import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    """Circular left shift by one."""
    # Your code here
    b= np.array(np.zeros(len(a)))
    for i in range(len(a)):
        b[i-1]=a[i]
    return b
