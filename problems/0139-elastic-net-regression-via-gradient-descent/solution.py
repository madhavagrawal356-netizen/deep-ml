import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:

    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for iteration in range(max_iter):
        # Calculate residuals
        e = X @ w + b - y

        # Calculate gradients
        grad_w = (X.T @ e) / n_samples + alpha1 * np.sign(w) + 2 * alpha2 * w
        grad_b = np.mean(e)

        # Update parameters
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        # Check convergence after update
        if np.linalg.norm(grad_w, ord=1) < tol:
            break

    return w, float(b)
    