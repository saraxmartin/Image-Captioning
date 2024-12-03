import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet201_Weights, ResNet50_Weights, VGG16_Weights



class EncoderCNN(nn.Module):
    def __init__(self, base_model, model_name, embed_size=1024):
        super(EncoderCNN, self).__init__()
        
        # Use the pretrained base model
        if model_name == "densenet201":
            self.base_model = base_model(pretrained=True, weights=DenseNet201_Weights.DEFAULT)
        elif model_name == "resnet50":
            self.base_model = base_model(pretrained=True, weights=ResNet50_Weights.DEFAULT)
        elif model_name == "vgg16":
            self.base_model = base_model(pretrained=True, weights=VGG16_Weights.DEFAULT)
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
            #-1 as it is Sequential module that contains multiple layers (last responsible producing the final predictions).
            in_features = self.base_model.classifier[-1].in_features 
            self.base_model.classifier = nn.Identity()  # Remove the final layer
        else:
            raise ValueError("Unsupported model")
        
        # Add additional layers
        self.embed = nn.Linear(in_features, embed_size)  # Fully connected embedding layer
        self.dropout = nn.Dropout(p=0.5)  # Dropout for regularization
        self.prelu = nn.PReLU()  # Parametric ReLU activation function

    def forward(self, images):
        # Extract features from the base model
        features = self.base_model(images)
        
        # Apply additional layers
        embeddings = self.embed(self.dropout(self.prelu(features)))
        return embeddings

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size=512):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size)
        self.attention = Attention(hidden_size, attention_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # word sequence
        print("Max caption index:", captions.max().item())
        print("Vocab size:", self.vocab_size)

        embedded_captions = self.embedding(captions)
        #print("embedded_captions vector:", embedded_captions.shape)

        
        # Process through LSTM
        context, _ = self.attention(features)  #get attention-weighted context
        #print("Context before unsqueeze:",context.shape)

        context = context.unsqueeze(1)
        #print("Context after unsqueeze:", context.shape) #[16] -> [16,1]

        context = context.unsqueeze(1) # [16,1] -> [16,1,1]
        #print("Context after unsqueeze:", context.shape)
        
        #print("Embedded size(1)", embedded_captions.size(1)) 
        context = context.expand(-1, embedded_captions.size(1), -1)
        #print("Context after expand:", context.shape)  # shape: [16, 11, 1]

        context = context.expand(-1, -1, 256)
        #print("Context after expand to hidden_size dim:", context.shape)


        lstm_input = torch.cat((embedded_captions, context), dim=2)  # Concatenate [16,11,256] -> [16,11,256+256]
        #print("Input LSTM size:", lstm_input.shape) #[16,11,512]
        #lstm_out = self.lstm(lstm_input)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        # Get predicted output word
        output = self.fc_out(lstm_out)
        #print("Output LSTM size:", lstm_out.shape)
        return output
    


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
        self.encoder = EncoderCNN(base_model, model_name, embed_size)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, attention_size) #with attention
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        
        return outputs
