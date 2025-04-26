import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    k: int = 6 # Num. of Gaussians in Positional Encoding
    d_model: int = 62 # Num. of Features
    seq_len: int = 10 # Sequence length
    n_temporal_heads: int = 5 # Num. of temporal heads
    n_channel_heads: int = 5 # Num. of channel heads
    dropout: float = 0.1
    n_layers: int = 5 # Number of Layers
    d_output_emb: int = 64 # Dimension of output embedding



class PositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Embedding = k x d_model
        self.embedding = nn.Parameter(torch.zeros([config.k, config.d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)

        # Positions = seq_len x k, intializes a <seq_len> dim. tensor, unsqueeze makes it <seq_len> x 1, 
        # repeat 1 over the rows and k times over the cols makes it <seq_len> x <k>
        self.positions = torch.tensor([i for i in range(config.seq_len)], requires_grad=False).unsqueeze(1).repeat(1, config.k)
        s = 0.0
        interval = config.seq_len / config.k
        mu = []
        # Mu List Incremented by Interval
        for _ in range(config.k):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        
        # Mu: 1 x k
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        # 1 x k
        self.sigma = nn.Parameter(torch.ones(config.k).unsqueeze(0))
        
    def normal_pdf(self, pos, mu, sigma):
        a = pos - mu # seq_len x k
        # torch.mul = a * a -> Element wise multiplication
        log_p = -1*torch.mul(a, a)/(2*(sigma**2)) - torch.log(sigma) # seq_len x k
        return torch.nn.functional.softmax(log_p, dim=1) # seq_len x k

    def forward(self, inputs):
        pdfs = self.normal_pdf(self.positions, self.mu, self.sigma) # seq_len x k
        pos_enc = torch.matmul(pdfs, self.embedding) # seq_len x k @ k x d_model => (seq_len, d_model)
        
        return inputs + pos_enc.unsqueeze(0).repeat(inputs.size(0), 1, 1) # B x seq_len x d_model + B x seq_len x d_model 
    

# Single Encoder Block/Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.temporal_attention = nn.MultiheadAttention(config.d_model, config.n_temporal_heads, batch_first=True)
        self.channel_attention = nn.MultiheadAttention(config.seq_len, config.n_channel_heads, batch_first=True)
        
        self.ln_1 = nn.LayerNorm(config.d_model)
        
        self.cnn_units = 1
        
        """
        BatchNorm2D works on 4D input of time Batch x channel x sequence x feature, wherein it computes the mean and variance for each channel
        over all the batches and activations and then normalizes each activation in the channel with the computed mean and variance.s

        Padding in Conv2d adds a row/column on all the 4 sides for each padding.
        """
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.cnn_units, (1, 1)),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, self.cnn_units, (3, 3), padding=1),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, 1, (5, 5), padding=2),
            nn.BatchNorm2d(1),
            nn.Dropout(config.dropout),
            nn.ReLU()
        )
        
        self.ln_2 = nn.LayerNorm(config.d_model)

    def forward(self, src, temporal_attn_mask, channel_attn_mask):
        # Temporal and channel attention's concatenated + src for residual connection, Layer Norm applied before attention
        # B x seq_len x d_model - src
        # After attention and attention norm - B x seq_len x d_model
        src_normalized = self.ln_1(src)
        src = src + self.temporal_attention(src_normalized, src_normalized, src_normalized, key_padding_mask=temporal_attn_mask)[0] 
        + self.channel_attention(src_normalized.transpose(-1, -2), src_normalized.transpose(-1, -2), src_normalized.transpose(-1, -2), key_padding_mask = channel_attn_mask)[0].transpose(-1, -2)
        
        # CNN + src for residual connection, Layer Norm Applied befor cnn
        # B x 1 x seq_len x d_model -> unsqueeze
        # 1 x 1 Kernel = B x 1 x seq_len x d_model
        # 3 x 3 Kernel = B x 1 x seq_len x d_model
        # 5 x 5 Kernel = B x 1 x seq_len x d_model

        src = src + self.cnn(self.ln_2(src).unsqueeze(dim=1)).squeeze(dim=1)
            
        return src

# Encoder (Represents all the encoder layers combined)
class TransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Different layers/blocks of transformer encoders
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.d_model)

    def forward(self, src, temporal_attn_mask, channel_attn_mask):
        for layer in self.layers:
            src = layer(src, temporal_attn_mask, channel_attn_mask)

        # Final Layer Norm
        src = self.ln_f(src)

        return src

# Passes the Input through positional encoding and Transformer Encoder
class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.pos_encoding = PositionalEncoding(config) # B, seq_len, d_model

        self.encoder = TransformerEncoder(config) # B, seq_len, d_model

    def forward(self, inputs, temporal_attn_mask, channel_attn_mask):
        encoded_inputs = self.pos_encoding(inputs)

        return self.encoder(encoded_inputs, temporal_attn_mask, channel_attn_mask)
    

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Output of the below is = B x seq_len x d_model
        self.transformer = Transformer(config)
        
        # Flatten (B x seq_len x d_model) to (B x seq_len*d_model) and passed to linear layer
        # Output of below is (B x d_output_emb)
        self.linear_proj = nn.Sequential(
            nn.Linear(config.seq_len * config.d_model, ((config.seq_len * config.d_model) // 2)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(((config.seq_len * config.d_model) // 2), config.d_output_emb),
            nn.ReLU()
        )
        
        
    def forward(self, inputs, temporal_attn_mask, channel_attn_mask):
        # Input: B x seq_len x d_model

        # Flatten to convert from B x seq_len x d_model to B*seq_len x d_model
        out = self.linear_proj(torch.flatten(self.transformer(inputs, temporal_attn_mask, channel_attn_mask), start_dim=1, end_dim=2))
        
        return out # B x d_output_emb