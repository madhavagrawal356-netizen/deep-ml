import math
import torch


class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k, v):
        # k, v: (batch, heads, seq, head_dim)
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def __len__(self):
        return 0 if self.k is None else self.k.size(2)

    def reset(self):
        self.k = None
        self.v = None
def attend_with_cache(q, k_new, v_new, cache):
    k, v = cache.append(k_new, v_new)
    head_dim = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    return torch.softmax(scores, dim=-1) @ v
    
