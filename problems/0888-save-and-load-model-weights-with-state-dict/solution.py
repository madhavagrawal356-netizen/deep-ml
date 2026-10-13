import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    # TODO: serialize src's state dict into a buffer, rewind, then load it into dst
    torch.save(src.state_dict(), 'model_weights.pth')
    dst.load_state_dict(torch.load('model_weights.pth'))
    return dst
