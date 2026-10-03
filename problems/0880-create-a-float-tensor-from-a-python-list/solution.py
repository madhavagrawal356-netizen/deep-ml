import torch

def to_float_tensor(values):
    # TODO: return a torch.float32 tensor built from `values`
    values=torch.tensor(values,dtype=torch.float32)
    return values
