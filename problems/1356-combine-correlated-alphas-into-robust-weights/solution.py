import numpy as np


def combine_alphas(mu: np.ndarray, cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """Mean-variance weights from shrunk covariance, normalised to sum to 1.

    Args:
        mu (np.ndarray): (p,) expected returns of each alpha.
        cov (np.ndarray): (p, p) covariance matrix of the alphas.
        shrinkage (float): intensity alpha in [0, 1].

    Returns:
        np.ndarray: (p,) weights summing to 1.
    """
    # Your code here
    p,  = mu.shape 
    u_bar = np.trace(cov)/p
    sigma = (1-shrinkage)*cov+shrinkage*u_bar*np.eye(p)
    w_raw = np.linalg.solve(sigma,mu)
    return w_raw/w_raw.sum()
