import numpy as np


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation function.

    Args:
        x (np.ndarray): 1-D series.
        nlags (int): highest lag to compute.

    Returns:
        np.ndarray: length nlags + 1, entry 0 equal to 1.0.
    """
    # Your code here
    x_bar = np.mean(x)
    summ = sum((x[i]-x_bar)**2 for i in range(len(x)))
    acff = []
    for k in range(nlags+1):
        acff.append(sum((x[i]-x_bar)*(x[i-k]-x_bar) for i in range(k,len(x)))/summ)
    return np.array(acff)


def ljung_box(x: np.ndarray, nlags: int) -> float:
    """Ljung-Box Q statistic over lags 1..nlags."""
    # Your code here
    n = x.shape[0]
    x=acf(x,nlags)
    q=n*(n+2)*sum(x[k]**2/(n-k) for k in range(1,nlags+1))
    return q
