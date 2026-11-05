import numpy as np


import numpy as np

def causal_zscore(x, window):
    x = np.asarray(x, dtype=float)
    result = np.full(len(x), np.nan)

    for t in range(window - 1, len(x)):
        w = x[t - window + 1:t + 1]
        mu = np.mean(w)
        sigma = np.std(w, ddof=0)

        result[t] = 0.0 if sigma == 0 else (x[t] - mu) / sigma

    return result


def has_lookahead(feature_fn, x):
    x = np.asarray(x, dtype=float)

    base = feature_fn(x.copy())

    perturbed = x.copy()
    perturbed[-1] += 1000.0

    new = feature_fn(perturbed)

    return not np.allclose(
        base[:-1], new[:-1], equal_nan=True
    )
    



