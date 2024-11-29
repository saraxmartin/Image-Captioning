import torch
import torch.nn as nn
from torchvision import models

class EncoderCNN(nn.Module):
    def __init__(self, base_model, embed_size=1024):
        super(EncoderCNN, self).__init__()
        
        # Use the pretrained base model
        self.base_model = base_model(pretrained=True)
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
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size)
        self.attention = Attention(hidden_size, attention_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, hidden_state=None, cell_state=None):
        # word sequence
        embedded_captions = self.embedding(captions)
        
        # Process through LSTM
        context, _ = self.attention(features, hidden_state)  # Get attention-weighted context
        lstm_input = torch.cat((embedded_captions, context), dim=1)  # Concatenate
        lstm_out, (hidden_state, cell_state) = self.lstm(lstm_input.unsqueeze(0), (hidden_state, cell_state))
        
        # Get predicted output word
        output = self.fc_out(lstm_out.squeeze(0))
        
        return output, hidden_state, cell_state
class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()
        
        # Attention parameters
        self.attn = nn.Linear(hidden_size, attention_size)
        self.context = nn.Linear(attention_size, 1)
        
    def forward(self, features, hidden_state):
        # Calculate attention scores based on features and hidden state
        attn_weights = torch.tanh(self.attn(features))
        attn_scores = self.context(attn_weights)
        
        # Normalize attention scores using softmax
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Get context vector based on attention weights
        context = torch.sum(attn_weights * features, dim=1)
        
        return context, attn_weights
    
class CaptioningModel(nn.Module):
    def __init__(self, base_model, embed_size, hidden_size, vocab_size, attention_size):
        super(CaptioningModel, self).__init__()
        self.encoder = EncoderCNN(base_model, embed_size)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, attention_size) #with attention
    
    def forward(self, images, captions, hidden_state=None, cell_state=None):
        features = self.encoder(images)
        outputs, hidden_state, cell_state = self.decoder(features, captions, hidden_state, cell_state)
        
        return outputs, hidden_state, cell_state
