import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model,num_heads,dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff),
       nn.GELU(),  nn.Dropout(dropout), nn.Linear(d_ff,d_model))

    def forward(self, x):
        # TODO: pre-LN attention sublayer, then pre-LN MLP sublayer, both with residual
        x = self.norm1(x)
        out, _ = self.attn(x,x,x)
        x=x+out
        out = self.norm2(x)
        out = self.mlp(out)
        return x+out
        
