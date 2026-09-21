import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in',
                      distribution: str = 'normal', seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    if mode == 'fan_in':
        fan = n_in
    elif mode == 'fan_out':
        fan = n_out
    elif mode == 'fan_avg':
        fan = (n_in + n_out) / 2
    else:
        raise ValueError("mode must be 'fan_in', 'fan_out', or 'fan_avg'")

    if distribution == 'normal':
        std = np.sqrt(2.0 / fan)
        return np.random.normal(0.0, std, size=(n_in, n_out))
    elif distribution == 'uniform':
        limit = np.sqrt(6.0 / fan)
        return np.random.uniform(-limit, limit, size=(n_in, n_out))
    else:
        raise ValueError("distribution must be 'normal' or 'uniform'")