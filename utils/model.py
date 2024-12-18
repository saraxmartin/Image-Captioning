import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models import DenseNet201_Weights, ResNet50_Weights, VGG16_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_device():
    from main import DEVICE  # Move the import here, inside the function
    return DEVICE


class EncoderCNN(nn.Module):
    def __init__(self, base_model, model_name, embed_size):
        super(EncoderCNN, self).__init__()
        # Use the pretrained base model
        if model_name == "densenet201":
            weights = DenseNet201_Weights.DEFAULT  
            self.base_model = base_model(weights=weights)
        elif model_name == "resnet50":
            weights = weights=ResNet50_Weights.DEFAULT
            self.base_model = base_model(weights=weights)
        elif model_name == "vgg16":
            weights=VGG16_Weights.DEFAULT
            self.base_model = base_model(weights=weights)
        else:
            raise Exception("MODEL NOT SUPORTED")
        # Identify and replace the final layer of the base model
        if isinstance(self.base_model, models.DenseNet):
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()  # Remove the final layer
            self.features = nn.Sequential(*list(self.base_model.features.children()))
            self.output_dim = in_features
            print(self.output_dim)
        elif isinstance(self.base_model, models.ResNet):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove the final layer
            self.features = nn.Sequential(*list(self.base_model.children())[:-2])
            self.output_dim = in_features
            print(self.output_dim)
        elif isinstance(self.base_model, models.VGG):
            # classifier[0] refers to the first fully connected layer after flattening the convolutional output
            # The input to this layer is 25088 features (flattened output from the last convolutional layer)
            in_features = self.base_model.classifier[0].in_features  # 512 for VGG16
            self.base_model.classifier = nn.Identity()  # Remove the final layer
            self.features = nn.Sequential(*list(self.base_model.features.children()))
            self.output_dim = 512  # VGG16 always outputs 512 channels
            print(self.output_dim)
        else:
            raise ValueError("Unsupported model")
        
        # For NO attention: Add additional layers
        self.embed = nn.Linear(self.output_dim, embed_size)  # Fully connected embedding layer
        self.dropout = nn.Dropout(p=0.5)  # Dropout for regularization
        self.prelu = nn.PReLU()  # Parametric ReLU activation function
        self.batch_norm = nn.BatchNorm1d(embed_size)  # Batch Normalization layer
        
        self.embed_size = embed_size


    def forward(self, images, attention=True):
        #print("\nENCODER:")

        if attention:
            # Pass images through sequential model.
            #print("0.Images shape: ",images.shape) # Input feature map [batch_size, num_channels(3), height(256), width(256)]
            features = self.features(images) #Output feature map [batch_size, num_channels(512), height(16), width(16)]
            #print("1.Features shape: ", features.shape)
            # Permute the tensor dimensions to [batch_size, height, width, channels]
            features = features.permute(0, 2, 3, 1)
            #print("2.Features shape: ", features.shape)  
            # Flatten the tensor along height and width dimensions to be used in a fully connected
            features = features.view(features.size(0), -1, features.size(-1))
            #print("3.Features shape: ", features.shape)
            features = self.embed(features)
            #print("4.Features shape: ",features.shape)
            
            return features 
        
        else:
            # Extract features from the base model
            #print("0.Images shape: ",images.shape)
            features = self.base_model(images)
            #print("1.Features shape: ", features.shape)
            # Apply global average pooling if output is 4D, like VGG
            #if features.dim() > 2: #this is in case last layer isnt removed, 
                #features = nn.functional.adaptive_avg_pool2d(features, (1, 1))  # Reduce to (batch_size, 512, 1, 1) ex: (16,512,1,1)
                #features = features.view(features.size(0), -1)  # Flatten to (batch_size, 512) ex (16,512)
            # Apply additional layers
            embeddings = self.embed(self.dropout(self.prelu(features)))
            #print("2.Embeddings shape: ", embeddings.shape)
            
            return embeddings

class Attention(nn.Module):
    def __init__(self, in_features, decom_space, ATTENTION_BRANCHES=1, dropout = 0.5):
        super(Attention, self).__init__()
        self.M = in_features  # Input dimension for combined input
        self.L = decom_space  # Dimension of attention space
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES
        self.dropout = nn.Dropout(dropout)

        # Linear layers for projecting input into query-key space
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # Project combined input
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES)
        )

    def forward(self, features, words):
        """
        Forward pass of Attention.
        features: Image features (batch_size, num_features, embed_dim)
        words: Word embeddings (batch_size, seq_len, embed_dim)
        """
        # Combine image features and words
        combined = torch.cat((features, words), dim=1)  # Concatenate along sequence dimension
        
        A = self.attention(combined)  # Attention weights
        A = F.softmax(A, dim=1)  # Apply softmax over sequence dimension

        # Weighted sum to compute context vector
        context = torch.matmul(A.permute(0, 2, 1), combined)  # (batch_size, ATTENTION_BRANCHES, embed_dim)
        return context, A


class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim, num_layers=1, dropout = 0.5):
        super(GRUDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim + hidden_dim, 
                          hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def decode_step(self, input_word, hidden_state, context_vector):
        """
        Decodes one step, combining word embeddings and attention context.
        """
        # Embed the input word
        input_word_embedding = self.embedding(input_word)  # (batch_size, 1, embedding_dim)
        
        # Concatenate the context vector with word embedding
        gru_input = torch.cat((input_word_embedding, context_vector), dim=-1)  # (batch_size, 1, embed_dim + hidden_dim)
        
        # GRU step
        gru_out, hidden_state = self.gru(gru_input, hidden_state)
        
        # Predict the next word
        output = self.fc(gru_out.squeeze(1))  # (batch_size, vocab_size)
        return output, hidden_state

class CaptioningModel_GRU(nn.Module):
    def __init__(self, base_model, model_name, embed_size, hidden_size, vocab_size, dataset):
        super(CaptioningModel_GRU, self).__init__()
        device = get_device()
        self.name = model_name
        self.encoder = EncoderCNN(base_model, model_name, embed_size).to(device)
        self.decoder = GRUDecoder(hidden_size, vocab_size, embed_size).to(device)
        self.attention = Attention(embed_size, hidden_size, ATTENTION_BRANCHES=1).to(device)
        self.dataset = dataset
        self.teacher_forcing_ratio = 0.5 # Teacher forcing probability

    def forward(self, images, captions, max_seq_length, mode="train"):
        device = get_device()
        features = self.encoder(images)  # Extract image features
        words = self.decoder.embedding(captions)
        
        h0, att_weights = self.attention(features,words)  # Initial hidden state from attention
        h0 = h0.permute(1, 0, 2)  # Adjust shape for GRU: (num_layers, batch_size, hidden_dim)
        
        batch_size = captions.size(0)
        vocab_size = self.decoder.fc.out_features
        
        # Initialize outputs to store predictions for each time step
        outputs = torch.zeros(batch_size, max_seq_length, vocab_size).to(device)
        
        # Start token input (assuming <SOS> token index is 1)
        input_word = captions[:, 0].unsqueeze(1)  # (batch_size, 1)
        hidden_state = h0

        for t in range(1, max_seq_length):  # Start from 1 since 0 is <SOS>
            # Decode one step
            context_vector = h0.squeeze(0).unsqueeze(1)
            output, hidden_state = self.decoder.decode_step(input_word, hidden_state,context_vector)
            
            # Store logits for this time step
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
            if use_teacher_forcing and mode == "train":
                input_word = captions[:, t].unsqueeze(1)  # Use ground-truth word
            else:
                _, predicted_word = torch.max(output, dim=1)  # Use predicted word
                input_word = predicted_word.unsqueeze(1)
        
        return outputs, att_weights

    def generate_captions(self, h0, max_seq_length):
        """
        Generate captions one word at a time (auto-regressive decoding).
        """
        device = get_device()
        batch_size = h0.size(1)  # h0.shape: (num_layers, batch_size, hidden_dim)
        outputs = torch.zeros((batch_size, max_seq_length, self.decoder.fc.out_features)).to(device)
        input_word = torch.ones((batch_size, 1), dtype=torch.long).to(device)  # Start token <SOS>
        hidden_state = h0

        for t in range(max_seq_length):
            output, hidden_state = self.decoder.decode_step(input_word, hidden_state)
            outputs[:, t, :] = output
            _, predicted_word = torch.max(output, dim=1)  # Get the most likely word
            input_word = predicted_word.unsqueeze(1)
        
        return outputs  # (batch_size, max_seq_length, vocab_size)

