import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    # Subtract maximum along specified dimension
    maxi = torch.max(t, dim=dim, keepdim=True)[0]

    # Exponentiate
    t = torch.exp(t - maxi)

    # Normalize
    t = t / torch.sum(t, dim=dim, keepdim=True)

    return t
