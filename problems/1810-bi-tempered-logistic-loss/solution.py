
import numpy as np

def log_t(x, t):
    x = np.asarray(x, dtype=float)
    if t == 1:
        return np.log(x)
    return (x**(1-t) - 1) / (1-t)


def exp_t(x, t):
    x = np.asarray(x, dtype=float)
    if t == 1:
        return np.exp(x)
    return np.maximum(1 + (1-t)*x, 0)**(1/(1-t))


def tempered_softmax(a, t):
    a = np.asarray(a, dtype=float)

    if t == 1:
        a_shift = a - np.max(a, axis=1, keepdims=True)
        e = np.exp(a_shift)
        return e / np.sum(e, axis=1, keepdims=True)

    lo = np.min(a, axis=1, keepdims=True)
    hi = np.max(a, axis=1, keepdims=True) + 1

    # Expand upper bound until sum <= 1
    while np.any(np.sum(exp_t(a - hi, t), axis=1, keepdims=True) > 1):
        hi = 2 * hi + 1

    for _ in range(200):
        lam = (lo + hi) / 2
        y = exp_t(a - lam, t)
        sums = np.sum(y, axis=1, keepdims=True)

        lo = np.where(sums > 1, lam, lo)
        hi = np.where(sums > 1, hi, lam)

        if np.max(np.abs(sums - 1)) < 1e-13:
            break

    lam = (lo + hi) / 2
    return exp_t(a - lam, t)


def bi_tempered_loss(labels, a, t1, t2):
    a = np.asarray(a, dtype=float)
    labels = np.asarray(labels, dtype=int)

    N, C = a.shape
    y_hat = tempered_softmax(a, t2)

    p_true = y_hat[np.arange(N), labels]

    term1 = -log_t(p_true, t1)
    term2 = (1 - np.sum(y_hat**(2-t1), axis=1)) / (2-t1)

    return term1 - term2

