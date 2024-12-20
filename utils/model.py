import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models import DenseNet201_Weights, ResNet50_Weights, VGG16_Weights
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
        
class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size, teacher_forcing):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # Word embedding layer: converts words into vectors of fixed size (embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # LSTM layer
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
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
            #print("outputs: ", outputs.shape)

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

            return outputs, att_weights

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
        #print(f"# Encoder linear: {encoder_proj.shape}")
        #print(f"# Decoder linear: {decoder_proj.shape}")

        # Compute  scores
        scores = torch.tanh(encoder_proj + decoder_proj)  # [batch_size, seq_len, attention_size]
        scores = self.energy(scores).squeeze(-1)  # [batch_size, seq_len]

        # Compute attention weights over encoder outputs (probabilities)
        attention_weights = self.softmax(scores)  # [batch_size, seq_len]

        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size] weighted sum of encoder outputs

        return context, attention_weights

class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim, num_layers=1):
        super(GRUDecoder, self).__init__()
        
        # Embedding layer for text input
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU Decoder
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, 
                          num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for generating word predictions
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def decode_step(self, input_word, hidden_state):
        """
        Decodes one step in the sequence and predicts the next word.
        """
        # Convert input_word to embedding (batch_size, 1, embed_size)
        input_word_embedding = self.embedding(input_word)
        
        # Pass through the GRU to get the next hidden state
        gru_out, hidden_state = self.gru(input_word_embedding, hidden_state)
        
        # Get logits for the next word
        output = self.fc(gru_out.squeeze(1))  # Output shape: (batch_size, vocab_size)
        
        return output, hidden_state
            

    def forward(self, captions, h0):
        device = get_device()
        # Move inputs to device
        captions = captions.to(device)
        h0 = h0.to(device)

        # Embed captions
        embeddings = self.embedding(captions)  # (batch_size, seq_len, embedding_dim)
        
        # Decode the embeddings
        outputs, _ = self.gru(embeddings, h0)  # outputs: (batch_size, seq_len, hidden_dim)
        
        # Generate word probabilities
        outputs = self.fc(outputs)  # (batch_size, seq_len, vocab_size)
        
        return outputs

class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim, num_layers=1):
        super(LSTMDecoder, self).__init__()
        
        # Embedding layer for text input
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
        
        # LSTM Decoder
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for generating word predictions
        self.fc = nn.Linear(hidden_dim, vocab_size)
        

    def forward(self, captions, h0, c0):
        device = get_device()
        # Move inputs to device
        captions = captions.to(device)
        h0 = h0.to(device)
        c0 = c0.to(device)

        # Embed captions
        embeddings = self.embedding(captions)  # (batch_size, seq_len, embedding_dim)
        
        # Decode the embeddings
        outputs, (hn, cn) = self.lstm(embeddings, (h0, c0))  # outputs: (batch_size, seq_len, hidden_dim)
        
        # Generate word probabilities
        outputs = self.fc(outputs)  # (batch_size, seq_len, vocab_size)
        
        return outputs, (hn, cn)

    
class DecoderLSTM_new(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size, teacher_forcing):
        super(DecoderLSTM_new, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.attention = AdditiveAttention(hidden_size, attention_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_outputs, captions):
        device = get_device()
        # encoder_outputs [batch_size, seq_len, hidden_size]
        # captions [batch_size, max_len]
        #print(f"[DECODER] Encoder outputs device: {encoder_outputs.device}")
        #print(f"[DECODER] Captions device: {captions.device}")
        batch_size, max_len = captions.size(0), captions.size(1)
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(device)
        #print(f"[DECODER] Outputs tensor initialized on device: {outputs.device}")

        # Initialize hidden and cell states for the LSTM
        h, c = torch.zeros(1, batch_size, self.hidden_size).to(device), \
               torch.zeros(1, batch_size, self.hidden_size).to(device)
        #print(f"[DECODER] LSTM hidden state device: {h.device}, {c.device}")

        # Start decoding with the <SOS> token
        inputs = captions[:, 0]  # [batch_size]

        for t in range(1, max_len):
            embedded_captions = self.embedding(inputs).unsqueeze(1).to(device)  # [batch_size, 1, embed_size]
            #print(f"[DECODER] Embedded captions device (step {t}): {embedded_captions.device}")
            #print("0.Embedded captions:", embedded_captions.shape)
            context, att_weights = self.attention(encoder_outputs, h.squeeze(0))  # [batch_size, hidden_size]
            #print(f"[DECODER] Context device (step {t}): {context.device}")
            #print("1.Context vector:",context.shape)
            #print("1.Attention weigths:", att_weights.shape)
            
            # Concatenate context vector with embedded input
            lstm_input = torch.cat((embedded_captions, context.unsqueeze(1)), dim=2)  # [batch_size, 1, embed_size + hidden_size]
            #print("3. LSTM input:", lstm_input.shape)

            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            #print(f"[DECODER] LSTM output device (step {t}): {lstm_out.device}")
            #print("3. LSTM output:", lstm_out.shape)
            #print("Hidden state: ", h.shape)

            # Generate output word scores
            output = self.fc_out(lstm_out.squeeze(1))  # [batch_size, vocab_size]
            #print("Output shape: ", output.shape)
            #print(f"[DECODER] Output logits device (step {t}): {output.device}")
            outputs[:, t, :] = output
            #print("OUTPUTS: ", outputs.shape)

            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            inputs = captions[:, t] if teacher_force else output.argmax(dim=1)

        return outputs, att_weights


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
    
class Attention(nn.Module):
    def __init__(self,in_features,decom_space,ATTENTION_BRANCHES=1):
        super(Attention, self).__init__()
        self.M = in_features #Input dimension of the Values NV vectors 
        self.L = decom_space # Dimension of Q(uery),K(eys) decomposition space
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES


        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

    def forward(self, x):

        # H feature vector matrix  # NV vectors x M dimensions
        H = x.squeeze(0)
        # Attention weights
        A = self.attention(H)  # NVxATTENTION_BRANCHES
        A = A.permute(0,2,1)  # ATTENTION_BRANCHESxNV
        A = F.softmax(A, dim=1)  # softmax over NV
        
        # Context Vector (Attention Aggregation)
        Z = torch.matmul(A, H)  # ATTENTION_BRANCHESxM 
        
        return Z, A
    
class CaptioningModel_GRU(nn.Module):
    def __init__(self, base_model, model_name, embed_size, hidden_size, vocab_size, dataset):
        super(CaptioningModel_GRU, self).__init__()
        device = get_device()
        self.name = model_name
        self.encoder = EncoderCNN(base_model, model_name, embed_size).to(device)
        self.decoder = GRUDecoder(hidden_size, vocab_size, embed_size).to(device)  # with attention
        self.attention = Attention(embed_size, hidden_size, ATTENTION_BRANCHES=1).to(device)
        self.dataset = dataset
        #self.attention = AoA_GatedAttention(hidden_size, 256).to(device)

    def forward(self, images, captions, max_seq_length, mode="train"):
        device = get_device()
        features = self.encoder(images)
        h0, att_weights = self.attention(features)
        h0 = h0.permute(1, 0, 2)  # Adjust for GRU input shape
        teacher_forcing_ratio = 0
        #if mode == "train":
            # Training mode with teacher forcing
            #outputs = self.decoder(captions, h0)
        if mode == "train":
            batch_size = captions.size(0)
            vocab_size = self.decoder.fc.out_features

            # Initialize outputs to store predictions for each time step
            outputs = torch.zeros(batch_size, max_seq_length, vocab_size).to(device)

            # Start token (<SOS>), assuming it's the first token in the vocabulary
            input_word = captions[:, 1].unsqueeze(1)  
            #input_word = torch.ones(batch_size, 1, dtype=torch.long).to(device) 

            hidden_state = h0
            for t in range(max_seq_length):
                # Decode one step
                output, hidden_state = self.decoder.decode_step(input_word, hidden_state)

                # Store logits for this time step
                outputs[:, t, :] = output

                # Decide whether to use teacher forcing or model prediction
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_word = captions[:, t].unsqueeze(1)  # Use ground truth word
                else:
                    _, predicted_word = torch.max(output, dim=1)
                    input_word = predicted_word.unsqueeze(1)  # Use predicted word as next input

        else:
            # Evaluation mode with auto-regressive decoding
            outputs = self.generate_captions(h0, max_seq_length)

        return outputs, att_weights


    def generate_captions(self, h0, max_seq_length):
        """
        Generate captions one word at a time (auto-regressive decoding).
        """
        device = get_device()
        batch_size = h0.size(1)  # h0.shape: (num_layers, batch_size, hidden_dim)

        # Initialize output tensor to store predicted logits
        outputs = torch.zeros((batch_size, max_seq_length, self.decoder.fc.out_features)).to(device)

        # Start token (assuming <SOS> token index is 1)
        input_word = torch.ones((batch_size, 1), dtype=torch.long).to(device)
        #print(self.dataset.idx2word[input_word.item()])

        hidden_state = h0
        for t in range(max_seq_length):
            # Decode one step
            output, hidden_state = self.decoder.decode_step(input_word, hidden_state)

            # Store logits for this time step
            outputs[:, t, :] = output

            # Get the most likely word for the next step
            _, predicted_word = torch.max(output, dim=1)
            input_word = predicted_word.unsqueeze(1)
        print("SHAPE", outputs.shape)

        return outputs  # Return logits (batch_size, max_seq_length, vocab_size)




class CaptioningModel_LSTM(nn.Module):
    def __init__(self, base_model, model_name, embed_size, hidden_size, vocab_size):
        super(CaptioningModel_LSTM, self).__init__()
        device = get_device()
        self.name = model_name
        self.encoder = EncoderCNN(base_model, model_name, embed_size).to(device)
        #print("ENCODER: ", self.encoder)
        self.decoder = LSTMDecoder(hidden_size, vocab_size, embed_size).to(device) #with attention
        #print("DECODER: ", self.decoder)
        self.attention = Attention(embed_size,hidden_size,ATTENTION_BRANCHES=1).to(device)

    def forward(self, images, captions):
        features = self.encoder(images)
        h0, att_weights = self.attention(features)
        h0 = h0.permute(1, 0, 2)
        batch_size, _, hidden_size = h0.size()
        c0 = torch.zeros_like(h0)  # Initialize cell state as zeros
        outputs, _ = self.decoder(captions, h0, c0)
       
        return outputs, att_weights

