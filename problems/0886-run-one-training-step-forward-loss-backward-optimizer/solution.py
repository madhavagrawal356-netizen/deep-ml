import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # TODO: build an SGD optimizer, run one full forward/loss/backward/step cycle,
    # and return the pre-update loss as a Python float.

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    return float(loss)

