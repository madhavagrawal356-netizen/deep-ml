
import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:

    n = data.shape[0]
    alphas = np.zeros(n)
    bias = 0.0

    # Compute kernel matrix
    if kernel == 'linear':
        K = data @ data.T

    elif kernel == 'rbf':
        sq_norms = np.sum(data**2, axis=1)
        dist = sq_norms[:, None] + sq_norms[None, :] - 2 * (data @ data.T)
        K = np.exp(-dist / (2 * sigma**2))

    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")

    # Deterministic Pegasos updates
    for t in range(1, iterations + 1):
        eta = 1.0 / (lambda_val * t)

        for i in range(n):
            g = np.sum(alphas * labels * K[:, i]) + bias

            if labels[i] * g < 1:
                alphas[i] += eta * (1 - lambda_val * alphas[i])
                bias += eta * labels[i]
            else:
                alphas[i] -= eta * lambda_val * alphas[i]

    return alphas.tolist(), float(bias)
