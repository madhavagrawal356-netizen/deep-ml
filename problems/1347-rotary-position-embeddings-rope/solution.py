import torch
import torch.nn as nn


def build_rope_cache(seq_len, head_dim, base=10000.0):
    # TODO: inv_freq = 1 / (base ** (arange(0, head_dim, 2) / head_dim))
    # TODO: theta = outer(arange(seq_len), inv_freq)  -> (seq_len, head_dim // 2)
    # TODO: return (cos(theta), sin(theta))
    inv_freq = torch.tensor(1/(base**(torch.arange(0,head_dim,2).float()/head_dim)))
    positions = torch.arange(seq_len).float()

    theta = torch.outer(positions, inv_freq)
    return torch.cos(theta),torch.sin(theta)


def apply_rope(x, cos, sin):
    # TODO: split x into halves x1, x2 along the last dim
    # TODO: rotated = cat(-x2, x1)
    # TODO: return x * cat(cos, cos) + rotated * cat(sin, sin)
    x1,x2 = torch.chunk(x,2, dim=-1)
    rotated = torch.cat((-x2,x1), dim=-1)
    return x*torch.cat((cos,cos), dim=-1)+rotated*torch.cat((sin,sin),dim=-1)
