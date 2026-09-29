import numpy as np

def xavier_initialization(shape: tuple, mode: str = 'uniform', seed: int = None) -> np.ndarray:
    n_in, n_out = shape
    rng = np.random.seed(seed)
    if mode == 'uniform':
        limit = np.sqrt(6 / (n_in + n_out))
        matrix = np.random.uniform(-limit, limit, size=(n_in, n_out))
    elif mode == 'normal':
        std = np.sqrt(2 / (n_in + n_out))
        matrix = np.random.normal(0, std, size=(n_in, n_out))
    else:
        raise ValueError(f"mode must be 'uniform' or 'normal', got {mode!r}")

    return matrix