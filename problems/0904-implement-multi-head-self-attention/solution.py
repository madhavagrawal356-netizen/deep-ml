import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        assert d_model%num_heads==0, f"d_model should be divisible by num_heads"
        self.q_proj = nn.Linear(d_model,d_model, bias=False)
        self.k_proj = nn.Linear(d_model,d_model, bias =False)
        self.v_proj = nn.Linear(d_model,d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.d_head = d_model//num_heads
        self.num_heads = num_heads

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        B,T,d_model = x.shape
        queries = self.q_proj(x).view(B,T,self.num_heads,self.d_head).transpose(1,2)
        keys = self.k_proj(x).view(B,T,self.num_heads,self.d_head).transpose(1,2)
        values = self.v_proj(x).view(B,T,self.num_heads,self.d_head).transpose(1,2)
        scaled_dot_prod = queries@keys.transpose(-1,-2)/self.d_head**0.5
        if mask is not None:
            scaled_dot_prod+=mask
        weights = torch.softmax(scaled_dot_prod, dim=-1)
        scaled_dot_prod=weights@values
        final = scaled_dot_prod.transpose(1,2).contiguous().view(B,T,self.num_heads*self.d_head)
        return self.out_proj(final)


        
