import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet201_Weights, ResNet50_Weights, VGG16_Weights

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
        elif isinstance(self.base_model, models.ResNet):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove the final layer
        elif isinstance(self.base_model, models.VGG):
            # classifier[0] refers to the first fully connected layer after flattening the convolutional output
            # The input to this layer is 25088 features (flattened output from the last convolutional layer)
            in_features = self.base_model.classifier[0].in_features  # 25088 for VGG16
            self.base_model.classifier = nn.Identity()  # Remove the final layer
        else:
            raise ValueError("Unsupported model")
        
        # Add additional layers
        #print("In features:", in_features)
        self.embed = nn.Linear(in_features, embed_size)  # Fully connected embedding layer
        self.dropout = nn.Dropout(p=0.5)  # Dropout for regularization
        self.prelu = nn.PReLU()  # Parametric ReLU activation function

    def forward(self, images):
        # Extract features from the base model
        features = self.base_model(images)
        # Apply global average pooling if output is 4D, like VGG
        if features.dim() > 2: #this is in case last layer isnt removed, 
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))  # Reduce to (batch_size, 512, 1, 1) ex: (16,512,1,1)
            features = features.view(features.size(0), -1)  # Flatten to (batch_size, 512) ex (16,512)
        # Apply additional layers
        #print((self.prelu(features)).size())
        embeddings = self.embed(self.dropout(self.prelu(features)))
        return embeddings

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size)
        self.attention = Attention(hidden_size, attention_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        teacher_forcing_ratio = 0.5 #fixed
        # word sequence
        #print("Max caption index:", captions.max().item())
        #print("Vocab size:", self.vocab_size)
        batch_size = captions.size(0)
        max_len = captions.size(1)  # Sequence length (including SOS/EOS tokens)
        vocab_size = self.vocab_size
        hidden_size = features.size(-1)

        # Prepare outputs tensor
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(features.device)
        embedded_captions = self.embedding(captions)
        #print("embedded_captions vector:", embedded_captions.shape)

        
        # Process through LSTM
        context, _ = self.attention(features)  #get attention-weighted context
        #print("Context before unsqueeze:",context.shape)

        context = context.unsqueeze(1)
        #print("Context after unsqueeze:", context.shape) #[16] -> [16,1]

        context = context.unsqueeze(1) # [16,1] -> [16,1,1]
        #print("Context after unsqueeze:", context.shape)
 
        # shape [batch_size, seq_len, hidden_dim]
        context = context.expand(-1, embedded_captions.size(1), embedded_captions.shape[-1]) 
        #print("Context after expand to hidden_size dim:", context.shape)

        inputs = captions[:, 0]  # Start with the <SOS> token
        pad_idx = 0
        sos_idx = 1
        eos_idx = 2
        unk_idx = 3
        for t in range(1, max_len):
            embedded_input = self.embedding(inputs).unsqueeze(1) # Shape: [batch_size, 1, embed_size]
            lstm_input = torch.cat((embedded_input, context[:, t - 1:t]), dim=2)  # Shape: [batch_size, 1, embed_size + hidden_size]
            #print("Input LSTM size:", lstm_input.shape) #[16,11,512]
            #lstm_out = self.lstm(lstm_input)
            lstm_out, (h_n, c_n) = self.lstm(lstm_input)
            # Get predicted output word
            output = self.fc_out(lstm_out)
            #print("Output LSTM size:", lstm_out.shape)
            output = output.squeeze(1)
            output[:, pad_idx] = float('-inf')  # Set logit for padding token to -inf
            output[:, sos_idx] = float('-inf') 
            output[:, eos_idx] = float('-inf') 
            output[:, unk_idx] = float('-inf')
            outputs[:, t, :] = output
            
            # Use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            inputs = captions[:, t] if teacher_force else output.argmax(dim=1)

            # Stop if we hit the EOS token in the ground truth sequence
            if (captions[:, t] == eos_idx).all():
                break

        return outputs
    


class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()
        
        # Attention parameters
        self.attn = nn.Linear(hidden_size, attention_size)
        self.context = nn.Linear(attention_size, 1)
        
    def forward(self, features):
        # Calculate attention scores based on features and hidden state
        attn_weights = torch.tanh(self.attn(features))
        attn_scores = self.context(attn_weights)
        
        # Normalize attention scores using softmax
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Get context vector based on attention weights
        context = torch.sum(attn_weights * features, dim=1)
        
        return context, attn_weights
    
class CaptioningModel(nn.Module):
    def __init__(self, base_model, model_name, embed_size, hidden_size, vocab_size, attention_size):
        super(CaptioningModel, self).__init__()
        self.name = model_name
        self.encoder = EncoderCNN(base_model, model_name, embed_size)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, attention_size) #with attention
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        
        return outputs
