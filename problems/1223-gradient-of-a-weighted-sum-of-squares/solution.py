import torch

def grad_wss(w_list, x_list):
    """Build w (requires_grad) and x from lists, compute
    loss = 0.5 * sum((w * x)**2), backward, return w.grad
    as a list of floats rounded to 4 decimals.
    """
    # TODo
    w = torch.tensor(w_list, requires_grad=True)
    x = torch.tensor(x_list, requires_grad=False)
    loss = 0.5*sum((w[i]*x[i])**2 for i in range(len(w)))
    loss.backward()
    return torch.round(w.grad, decimals=4).tolist()
