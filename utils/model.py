import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet201_Weights, ResNet50_Weights, VGG16_Weights
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        self.embed = nn.Linear(self.output_dim, embed_size).to(DEVICE)  # Fully connected embedding layer

    def forward(self, images, attention=True):
        #print("\nENCODER:")
        images = images.to(DEVICE)
        # Pass images through sequential model.
        #print("0.Images shape: ",images.shape) # Input feature map [batch_size, num_channels(3), height(256), width(256)]
        features = self.features(images).to(DEVICE) #Output feature map [batch_size, num_channels(512), height(16), width(16)]
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
        
class FirstDecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size, num_layers=1):
        super(FirstDecoderLSTM, self).__init__()
        # Word embedding layer: converts words into vectors of fixed size (embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size).to(DEVICE)
        # LSTM layer
        #self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True,
        #                    num_layers=num_layers, dropout=0.5 if num_layers > 1 else 0).to(DEVICE)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True,
                            num_layers=num_layers, dropout=0.5 if num_layers > 1 else 0).to(DEVICE)
        # Attention layer
        #self.attention = AoA_GatedAttention(hidden_size, attention_size).to(DEVICE)
        self.attention = FirstAdditiveAttention(hidden_size, attention_size).to(DEVICE)
        # Linear layer: maps output of LSTM into the size of the vocabulary
        # Gives scores for each word in the vocabulary being the next word in the caption 
        self.fc_out = nn.Linear(hidden_size, vocab_size).to(DEVICE)
        # Dropout layer
        self.dropout= nn.Dropout(p=0.5).to(DEVICE)

    def forward(self, features, captions, type):
        features, captions = features.to(DEVICE), captions.to(DEVICE)

        if type=="train":
            embedded_captions = self.dropout(self.embedding(captions))
        elif type=="val" or type=="test":
            embedded_captions = self.embedding(captions)
        
        # AOA attention
        #context, att_weights = self.attention(features)
        #context = context.unsqueeze(1).repeat(1, embedded_captions.size(1), 1)
        # Additive attention
        att_weights = self.attention(features, features.mean(dim=1))  # We use the mean features as the initial hidden state
        context = torch.sum(features * att_weights.unsqueeze(2), dim=1)  # (batch_size, hidden_size)
        context = context.unsqueeze(1).repeat(1, embedded_captions.size(1), 1)
        
        lstm_input = torch.cat((context,embedded_captions), dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        outputs = self.fc_out(self.dropout(lstm_out))

        return outputs

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size, teacher_forcing):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing

        self.embedding = nn.Embedding(vocab_size, embed_size).to(DEVICE)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True).to(DEVICE)
        self.attention = AdditiveAttention(hidden_size, attention_size).to(DEVICE)
        self.fc_out = nn.Linear(hidden_size, vocab_size).to(DEVICE)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, encoder_outputs, captions, type):

        # Move inputs to the correct device
        encoder_outputs = encoder_outputs.to(DEVICE)
        captions = captions.to(DEVICE)

        if type == "val" or type == "test":
            self.teacher_forcing_ratio = 0

        batch_size, max_len = captions.size(0), captions.size(1)
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(DEVICE)

        # Initialize hidden and cell states for the LSTM
        h, c = torch.zeros(1, batch_size, self.hidden_size).to(DEVICE), \
               torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)

        # Start decoding with the <SOS> token
        inputs = captions[:, 0]  # [batch_size]

        for t in range(1, max_len):
            embedded_captions = self.embedding(inputs).unsqueeze(1)  # [batch_size, 1, embed_size]
            #print("0.Embedded captions:", embedded_captions.shape)
            context, att_weights = self.attention(encoder_outputs, h.squeeze(0))  # [batch_size, hidden_size]
            #print("1.Context vector:",context.shape)
            #print("1.Attention weigths:", att_weights.shape)
            
            # Concatenate context vector with embedded input
            lstm_input = torch.cat((embedded_captions, context.unsqueeze(1)), dim=2)  # [batch_size, 1, embed_size + hidden_size]
            #print("3. LSTM input:", lstm_input.shape)

            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            lstm_out = self.dropout(lstm_out)
            #print("3. LSTM output:", lstm_out.shape)
            #print("Hidden state: ", h.shape)

            # Generate output word scores
            output = self.fc_out(lstm_out.squeeze(1))  # [batch_size, vocab_size]
            #print("Output shape: ", output.shape)
            outputs[:, t, :] = output
            #print("OUTPUTS: ", outputs.shape)

            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            inputs = captions[:, t] if teacher_force else output.argmax(dim=1)

        return outputs

class DecoderLSTM_noAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, teacher_forcing):
        super(DecoderLSTM_noAttention, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing

        self.embedding = nn.Embedding(vocab_size, embed_size).to(DEVICE)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True).to(DEVICE)
        self.fc_out = nn.Linear(hidden_size, vocab_size).to(DEVICE)

        # Fully connected layer to initialize hidden states from encoder outputs
        self.init_h = nn.Linear(hidden_size, hidden_size).to(DEVICE)
        self.init_c = nn.Linear(hidden_size, hidden_size).to(DEVICE)

    def forward(self, encoder_outputs, captions, type):
        # Move inputs to the correct device
        encoder_outputs = encoder_outputs.to(DEVICE)
        captions = captions.to(DEVICE)

        if type == "val" or type == "test":
            self.teacher_forcing_ratio = 0

        batch_size, max_len = captions.size(0), captions.size(1)
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(DEVICE)

        # Compute the initial hidden and cell states from encoder outputs (e.g., mean of encoder outputs)
        encoder_summary = encoder_outputs.mean(dim=1)  # [batch_size, hidden_size]
        h = self.init_h(encoder_summary).unsqueeze(0)  # [1, batch_size, hidden_size]
        c = self.init_c(encoder_summary).unsqueeze(0)  # [1, batch_size, hidden_size]

        # Start decoding with the <SOS> token
        inputs = captions[:, 0]  # [batch_size]

        for t in range(1, max_len):
            embedded_captions = self.embedding(inputs).unsqueeze(1)  # [batch_size, 1, embed_size]
            
            # LSTM step without attention
            lstm_out, (h, c) = self.lstm(embedded_captions, (h, c))

            # Generate output word scores
            output = self.fc_out(lstm_out.squeeze(1))  # [batch_size, vocab_size]
            outputs[:, t, :] = output

            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            inputs = captions[:, t] if teacher_force else output.argmax(dim=1)

        return outputs


class FirstAdditiveAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(FirstAdditiveAttention, self).__init__()

        #self.project = nn.Linear(hidden_size,attention_size).to(DEVICE)
        self.attention_layer = nn.Linear(hidden_size + hidden_size, 1).to(DEVICE)

    def forward(self, encoder_out, h_prev): 
        # Project encoder outputs and decoder hidden state to attention_size
        #encoder_out = self.project(encoder_out).to(DEVICE)  # [batch_size, seq_len, attention_size]
        #h_prev = self.project(h_prev).unsqueeze(1).to(DEVICE)  # [batch_size, 1, attention_size]
        #we want to have a prev_h that matches the size of encoder_out to be able to concat them
        h_prev_repeated = h_prev.unsqueeze(1).repeat(1, encoder_out.shape[1], 1) 

        #we concattenate both tensors to have an nput for the fully connected layer which will then output the attention weights
        att_input = torch.cat((encoder_out, h_prev_repeated), dim=2)                                                          
        
        #we get the attention scores for the "pixel" we're looking at each step
        att_scores = self.attention_layer(att_input) # [32, 196, 1][batch_size, num_pixels, 1]

        #Get rid of the last dimension
        att_scores = att_scores.squeeze(2) # [32, 196][batch_size, num_pixels]

        #Make all the scores sum up to 1 via softmax, hence we'll get the weights
        att_weights = F.softmax(att_scores, dim=1) #[32, 196][batch_size, num_pixels]

        return att_weights #[32, 196][batch_size, num_pixels]
    
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(AdditiveAttention, self).__init__()
        self.encoder_proj = nn.Linear(hidden_size, attention_size).to(DEVICE)
        self.decoder_proj = nn.Linear(hidden_size, attention_size).to(DEVICE)
        self.energy = nn.Linear(attention_size, 1).to(DEVICE)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs, decoder_hidden):
        # Move inputs to the correct device
        encoder_outputs = encoder_outputs.to(DEVICE)
        decoder_hidden = decoder_hidden.to(DEVICE)
        # Project encoder outputs and decoder hidden state to attention_size
        encoder_proj = self.encoder_proj(encoder_outputs).to(DEVICE)  # [batch_size, seq_len, attention_size]
        decoder_proj = self.decoder_proj(decoder_hidden).unsqueeze(1).to(DEVICE)  # [batch_size, 1, attention_size]
        #print(f"# Encoder linear: {encoder_proj.shape}")
        #print(f"# Decoder linear: {decoder_proj.shape}")

        # Compute  scores
        scores = torch.tanh(encoder_proj + decoder_proj)  # [batch_size, seq_len, attention_size]
        scores = self.energy(scores).squeeze(-1).to(DEVICE)  # [batch_size, seq_len]

        # Compute attention weights over encoder outputs (probabilities)
        attention_weights = self.softmax(scores).to(DEVICE)  # [batch_size, seq_len]

        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1).to(DEVICE)  # [batch_size, hidden_size] weighted sum of encoder outputs

        return context, attention_weights
class AoA_GatedAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(AoA_GatedAttention, self).__init__()
        
        # First attention layer: standard attention mechanism
        # Linear layer that projects input features from hidden_size (512) to attention size (256)
        self.attn = nn.Linear(hidden_size, attention_size)
        # Linear layer that computes attention scores based on attention weights
        self.context = nn.Linear(attention_size, 1)
        
        # Second attention layer: Attention on Attention
        # Linear layer to refine attention weigths
        self.attn_on_attn = nn.Linear(1, attention_size)
        # Linear layer to compute refined attention scores from attn on attn mechanism
        self.context_on_attn = nn.Linear(attention_size, 1)
        
        # Gating mechanism
        self.gate = nn.Sigmoid()  # Sigmoid gate to refine attention weights
        
    def forward(self, features):
        # Step 1: Apply initial attention
        attn_scores = torch.tanh(self.attn(features)) #[batch_size, 256, attention_size]
        attn_scores = self.context(attn_scores) #[batch_size, 256, 1]
        
        # Normalize attention scores using softmax
        #attn_weights = torch.softmax(attn_scores, dim=1) #[batch_size, 256, 1]
        
        # Step 2: Apply AoA (Attention on Attention)
        attn_weights_on_weights = torch.tanh(self.attn_on_attn(attn_scores))
        refined_attn_scores = self.context_on_attn(attn_weights_on_weights)
        
        # Normalize refined attention scores
        refined_attn_weights = torch.softmax(refined_attn_scores, dim=1)
        
        # Step 3: Apply gating mechanism to refine attention weights
        gated_attn_weights = refined_attn_weights * self.gate(refined_attn_weights)  #[batch_size, 256, 1]
        
        # Get context vector based on gated attention weights
        context = torch.sum(gated_attn_weights * features, dim=1) #[batch_size, 512]
        
        return context, gated_attn_weights

class CaptioningModel(nn.Module):
    def __init__(self, base_model, model_name, embed_size, hidden_size, vocab_size, attention_size, method=2):
        super(CaptioningModel, self).__init__()
        self.name = model_name
        self.encoder = EncoderCNN(base_model, model_name, embed_size).to(DEVICE)
        if method == 1: # Overfitting
            self.decoder = FirstDecoderLSTM(embed_size, hidden_size, vocab_size, attention_size).to(DEVICE)
        if method == 2:
            self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, attention_size, teacher_forcing=0.7).to(DEVICE)
        if method == 3:
            self.decoder = DecoderLSTM_noAttention(embed_size, hidden_size, vocab_size, teacher_forcing=0.7)

    def forward(self, images, captions, type):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, type)
        
        return outputs
