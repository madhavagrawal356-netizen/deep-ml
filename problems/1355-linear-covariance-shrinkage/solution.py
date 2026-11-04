import numpy as np


def shrink_covariance(returns: np.ndarray, shrinkage: float) -> np.ndarray:
    """Linear shrinkage of the sample covariance toward a scaled identity.

    Args:
        returns (np.ndarray): (n_samples, n_assets) return matrix.
        shrinkage (float): intensity alpha in [0, 1].

    Returns:
        np.ndarray: (n_assets, n_assets) shrunk covariance matrix.
    """
    # Your code here
    _, p = returns.shape
    S = np.cov(returns, rowvar=False, ddof=1)
    F = np.trace(S)/p*np.eye(p)
    return (1-shrinkage)*S+shrinkage*F


def min_eigenvalue(matrix: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix."""
    return np.linalg.eigvalsh(matrix)[0]
