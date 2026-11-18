import numpy as np

def make_diagonal(x):
    # Your code here
    n = x.shape[0]
    d = np.zeros((n,n))
    for i in range(len(x)):
        d[i][i]=x[i]
    return d