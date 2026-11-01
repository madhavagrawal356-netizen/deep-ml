import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """Reverse 1-D array a without slicing a[::-1]."""
    n = len(a)
    indices = np.arange(n-1, -1, -1)
    result = a[indices]
    return result
    
