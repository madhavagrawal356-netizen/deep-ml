import numpy as np


def logit_margin(z, t):
    """Margin m = z_t - log sum_{j != t} exp(z_j), computed stably; returns a Python float."""
    
    z2 = np.array(z)
    z2 = np.delete(z2,t)
    p = np.argmax(z2)
    s = sum(np.exp(z2[i]-z2[p]) for i in range(len(z2)))
    return float(z[t]-np.log(s)-z2[p])


def precision_cliff(dtype):
    """Smallest m >= 0 at which 1/(1 + exp(-m)) == 1 when evaluated in dtype; bisection on a float64 m.

    Returns the margin rounded to 4 decimals.
    """
    left = 0.0000
    right = 100.0000
    while right-left>10**-9:
        a = (left+right)/2
        if dtype(np.exp(-a))+dtype(1)>dtype(1):
            left=a 
        elif dtype(np.exp(-a))+dtype(1)==dtype(1):
            right=a 
        else:
            right=a 
    return float(np.round(a,4))


