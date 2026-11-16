import torch

def train_softmaxreg(X: torch.Tensor, y: torch.Tensor,
                     learning_rate: float, iterations: int) -> tuple:
    """
    Gradient-descent training algorithm for Softmax regression,
    optimizing parameters with Cross Entropy loss.
    """
    N, D = X.shape
    C = int(torch.max(y).item()) + 1

    # Add bias column as the first column
    X = torch.cat((torch.ones(N, 1, dtype=X.dtype, device=X.device), X), dim=1)

    # One-hot encode labels
    Y = torch.nn.functional.one_hot(y.long(), num_classes=C).to(X.dtype)

    # Initialize coefficients to zero: C x (D+1)
    W = torch.zeros(C, D + 1, dtype=X.dtype, device=X.device)

    losses = []

    for _ in range(iterations):
        # Compute logits and softmax probabilities
        logits = X @ W.T
        probs = torch.softmax(logits, dim=1)

        # Sum-based cross entropy loss (before weight update)
        loss = -torch.sum(Y * torch.log(probs + 1e-15))
        losses.append(round(loss.item(), 4))

        # Sum-based gradient
        gradient = (probs - Y).T @ X

        # Gradient descent update
        W -= learning_rate * gradient

    # Return rounded coefficients and losses
    coefficients = torch.round(W * 10000) / 10000
    coefficients = coefficients.tolist()

    return coefficients, losses
