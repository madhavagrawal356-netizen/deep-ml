import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    # Your code here
    n=len(y)
    if replacements:
        subset_size = n 
    else:
        subset_size=n//2
    result=[]
    for subset in range(n_subsets):
        indices = np.random.choice(n,subset_size, replace=replacements)
        X_part = X[indices,:]
        y_part = y[indices]
        result.append((X_part,y_part))
    return result