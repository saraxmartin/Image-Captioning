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
            self.features = nn.Sequential(*list(self.base_model.features.children()))
        elif isinstance(self.base_model, models.ResNet):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove the final layer
            self.features = nn.Sequential(*list(self.base_model.children())[:-2])
        elif isinstance(self.base_model, models.VGG):
            # classifier[0] refers to the first fully connected layer after flattening the convolutional output
            # The input to this layer is 25088 features (flattened output from the last convolutional layer)
            in_features = self.base_model.classifier[0].in_features  # 512 for VGG16
            self.base_model.classifier = nn.Identity()  # Remove the final layer
            self.features = nn.Sequential(*list(self.base_model.features.children()))
        else:
            raise ValueError("Unsupported model")
        
        # For NO attention: Add additional layers
        self.embed = nn.Linear(in_features, embed_size)  # Fully connected embedding layer
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
            embed_layer = nn.Linear(features.size(2), self.embed_size) # Linear embedding to get equal dim for all backbones
            features = embed_layer(features)
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
        
class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size, teacher_forcing):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # Word embedding layer: converts words into vectors of fixed size (embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # LSTM layer
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size)
        # Attention layer
        self.attention = AoA_GatedAttention(hidden_size, attention_size)
        # Linear layer: maps output of LSTM into the size of the vocabulary
        # Gives scores for each word in the vocabulary being the next word in the caption 
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        # Teacher forcing ratio (if 0: no teacher forcing)
        self.teacher_forcing_ratio = teacher_forcing

    def forward(self, features, captions, attention=True):
        #print("\nDECODER:")

        # Apply word embeddings to the captions
        embedded_captions = self.embedding(captions)
        #print("0.Embedded captions:", embedded_captions.shape)

        # Initialize LSTM hidden states
        #outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(features.device)
        #print("0.Outputs preparation: ",outputs.shape)

        if attention:

            # Get attention-weighted context
            context, att_weights = self.attention(features)
            #print("1.Context vector:",context.shape)
            #print("1.Attention weigths:", att_weights.shape)

            # Expand context vector with the word embeddings size
            context = context.unsqueeze(1).repeat(1, embedded_captions.size(1), 1)
            #print("2. Context after expanding to seq len size:", context.shape)

            # Concatenate with word embeddings and input to LSTM
            lstm_input = torch.cat((context,embedded_captions), dim=2)
            #print("3. LSTM input:", lstm_input.shape)
            lstm_out, _ = self.lstm(lstm_input)
            #print("3. LSTM output:", lstm_out.shape)

            # Pass the LSTM output through the linear layer to get the output scores for each word in the vocabulary
            outputs = self.fc_out(lstm_out)

            """inputs = captions[:, 0]  # Start with the <SOS> token
            pad_idx = 0
            sos_idx = 1
            eos_idx = 2
            unk_idx = 3
            for t in range(1, seq_len):
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
                teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
                inputs = captions[:, t] if teacher_force else output.argmax(dim=1)

                # Stop if we hit the EOS token in the ground truth sequence
                if (captions[:, t] == eos_idx).all():
                    break"""

            return outputs

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(AdditiveAttention, self).__init__()
        self.encoder_proj = nn.Linear(hidden_size, attention_size)
        self.decoder_proj = nn.Linear(hidden_size, attention_size)
        self.energy = nn.Linear(attention_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs, decoder_hidden):
        # Project encoder outputs and decoder hidden state to attention_size
        encoder_proj = self.encoder_proj(encoder_outputs)  # [batch_size, seq_len, attention_size]
        decoder_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)  # [batch_size, 1, attention_size]
        print(f"Encoder linear: {encoder_proj.shape}")
        print(f"Decoder linear: {encoder_proj.shape}")

        # Compute  scores
        scores = torch.tanh(encoder_proj + decoder_proj)  # [batch_size, seq_len, attention_size]
        scores = self.energy(scores).squeeze(-1)  # [batch_size, seq_len]

        # Compute attention weights over encoder outputs (probabilities)
        attention_weights = self.softmax(scores)  # [batch_size, seq_len]

        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size] weighted sum of encoder outputs

        return context, attention_weights

class DecoderLSTM_new(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size, teacher_forcing):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size)
        self.attention = AdditiveAttention(hidden_size, attention_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_outputs, captions):
        # encoder_outputs [batch_size, seq_len, hidden_size]
        # captions [batch_size, max_len]

        batch_size, max_len = captions.size(0), captions.size(1)
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(encoder_outputs.device)

        # Initialize hidden and cell states for the LSTM
        h, c = torch.zeros(1, batch_size, self.hidden_size).to(encoder_outputs.device), \
               torch.zeros(1, batch_size, self.hidden_size).to(encoder_outputs.device)

        # Start decoding with the <SOS> token
        inputs = captions[:, 0]  # [batch_size]

        for t in range(1, max_len):
            embedded_captions = self.embedding(inputs).unsqueeze(1)  # [batch_size, 1, embed_size]
            print("0.Embedded captions:", embedded_captions.shape)
            context, att_weights = self.attention(encoder_outputs, h.squeeze(0))  # [batch_size, hidden_size]
            print("1.Context vector:",context.shape)
            print("1.Attention weigths:", att_weights.shape)
            
            # Concatenate context vector with embedded input
            lstm_input = torch.cat((embedded_captions, context.unsqueeze(1)), dim=2)  # [batch_size, 1, embed_size + hidden_size]
            print("3. LSTM input:", lstm_input.shape)

            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            print("3. LSTM output:", lstm_out.shape)
            print("Hidden state: ", h.shape, h)

            # Generate output word scores
            output = self.fc_out(h.squeeze(0))  # [batch_size, vocab_size]
            outputs[:, t, :] = output

            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            inputs = captions[:, t] if teacher_force else output.argmax(dim=1)

        return outputs


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
        attn_weights = torch.tanh(self.attn(features)) #[batch_size, 256, attention_size]
        attn_scores = self.context(attn_weights) #[batch_size, 256, 1]
        
        # Normalize attention scores using softmax
        attn_weights = torch.softmax(attn_scores, dim=1) #[batch_size, 256, 1]
        
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
    def __init__(self, base_model, model_name, embed_size, hidden_size, vocab_size, attention_size):
        super(CaptioningModel, self).__init__()
        self.name = model_name
        self.encoder = EncoderCNN(base_model, model_name, embed_size)
        #print("ENCODER: ", self.encoder)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, attention_size, teacher_forcing=0.5) #with attention
        #print("DECODER: ", self.decoder)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        
        return outputs
