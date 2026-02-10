import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_modalities: int = 3 # Number of modalities
    raw_d_model: int = 55 # Num. of features in raw dataset
    d_model: int = 128 # Num. of Features
    seq_len: int = 200 # Sequence length
    n_temporal_heads: int = 4 # Num. of temporal heads
    n_layers: int = 5 # Number of Layers
    dropout: float = 0.1
    n_users: int = 79 # Number of users to classify
    contrastive_loss_alpha: int = 2 # Contrastive loss importance - hyperparameter (Alpha)



# MLP in transformer encoder
class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model) # Linear projection
        self.c_proj.SCALE_RES_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
# Single Encoder Block/Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.d_model)        
        self.temporal_attention = nn.MultiheadAttention(config.d_model, config.n_temporal_heads, batch_first=True, dropout=config.dropout) 

        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, src):
        # Temporal and channel attention's concatenated + src for residual connection, Layer Norm applied before attention
        # B x seq_len x d_model - src
        # After attention and attention norm - B x seq_len x d_model
        src_normalized = self.ln_1(src)

        src = src + self.temporal_attention(src_normalized, src_normalized, src_normalized)[0]
        # src = self.ln_1(src)
        src = src + self.mlp(self.ln_2(src))
        # src = self.ln_2(src)
        
        return src

# Encoder (Represents all the encoder layers combined)
class TransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Different layers/blocks of transformer encoders
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.d_model)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
            # print(f"After Transformer Block: {src.mean()} | std: {src.std()}")

        # Final Layer Norm
        src = self.ln_f(src)
        
        # print(f"After Encoder mean: {src.mean()} | std: {src.std()}")

        return src

# Passes the Input through positional encoding and Transformer Encoder
class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Normal pos. emb, +1 for CLS token
        self.pos_encoding = nn.Parameter(torch.randn(1, config.seq_len, config.d_model)) # (1, seq_len + 1, d_model)

        self.encoder = TransformerEncoder(config) # B, seq_len, d_model

    def forward(self, inputs):

        # Normal pos. emb
        encoded_inputs = inputs + self.pos_encoding[:, :inputs.shape[1], :] # (B, seq_len + 1, d_model) + (1, seq_len + 1, d_model) => (B, seq_len+1, d_model)

        return self.encoder(encoded_inputs)
    

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config

        self.input_norm = nn.LayerNorm(config.raw_d_model)

        # Linear layer for converting raw feature dimension to required feature dimension
        self.linear_proj_1 = nn.Linear(config.raw_d_model, config.d_model)

        self.ln = nn.LayerNorm(config.d_model)

        # Modality projection
        self.modality_proj = nn.Linear(config.n_modalities, config.d_model)

        # Output of the below is = B x seq_len x d_model
        self.transformer = Transformer(config)
        
        self.final_proj = nn.Sequential(
            nn.Linear(config.seq_len * config.d_model, (config.seq_len * config.d_model) // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear((config.seq_len * config.d_model) // 2, config.d_model),
            nn.ReLU()
        )

        self.classifier = nn.Linear(config.d_model, config.n_users, bias=False) # Final Classification layer
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'SCALE_RES_INIT'):
                std = module.weight.std().item()
                std *= (2 * self.config.n_layers)**-0.5 # Std. scaled by a factor of the number of residual layers, at the end of MLP
                torch.nn.init.normal_(module.weight, mean=0, std=std )

            # Initalizing linear layer biases to 0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        if isinstance(module, nn.MultiheadAttention):
            std = module.out_proj.weight.std().item() # Current std
            std *= (2 * self.config.n_layers)**-0.5 # Std. scaled by a factor of the number of residual layers
            torch.nn.init.normal_(module.out_proj.weight, mean=0, std=std)

             # Initializing biases to 0
            if module.out_proj.bias is not None:
                torch.nn.init.zeros_(module.out_proj.bias)
            if module.in_proj_bias is not None:
                torch.nn.init.zeros_(module.in_proj_bias)
  
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that requires grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available - when running on CUDA
        # Instead of iterating over all the tensors and updating them which would launch many kernels, Fused would fuse all these kernels 
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=use_fused)

        return optimizer
    
    def forward(self, inputs, modality_mask, targets=None):
        # Input: (B, seq_len,  raw_d_model)
        # Modality Mask: (B, seq_len, n_modalities)

        inputs = self.input_norm(inputs)

        # Linear project to d_model
        inputs = self.linear_proj_1(inputs) # (B, seq_len, d_model)
        modality_embed = self.modality_proj(modality_mask.float()) # (B, seq_len, n_modalities) @ (n_modalities, d_model) => (B, seq_len, d_model)
        inputs = inputs + modality_embed # Adding the modality embedding to the inputs
        inputs = self.ln(inputs)

        # Pass to the transformer
        out = self.transformer(inputs) # (B, seq_len, d_model)
        out = self.final_proj(torch.flatten(out, start_dim=1, end_dim=2))
        
        # print(f"After Linear Projection mean: {out.mean()} | std: {out.std()}")

        # B x n_users
        logits = self.classifier(out)

        # print(f"After Logits mean: {logits.mean()} | std: {logits.std()}")
        
        # Loss initialized as None
        loss = None
        cos_loss = None
        cross_entropy_loss = None

        if targets is not None:
            
            # Cross entropy loss
            loss = F.cross_entropy(logits, targets)

        # Return embeddings, logits and loss
        return out, logits, loss 
    

if __name__ == "__main__":
    model_config = ModelConfig(
        n_modalities = 2,
        raw_d_model = 46, # Num. of features in raw dataset
        d_model= 64, # Num. of features
        seq_len= 200, # Block size/seq. length
        n_temporal_heads= 4, # Num. of temporal heads
        dropout=0.4,
        n_layers= 5, # Number of layers or transformer encoders
        n_users = 79, # Number of users (For classification)
        contrastive_loss_alpha = 1 # Contrastive loss importance hyperparameter (Alpha)
    )
    model = Model(model_config)
    num_of_parameters = sum(p.numel() for p in model.parameters())
    print("Num. of parameters", num_of_parameters)
    print("****************************")
    for name, _ in model.named_parameters():
        print("Name", name)

