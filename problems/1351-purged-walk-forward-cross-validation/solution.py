import numpy as np


def purged_walk_forward_splits(n_samples: int, n_splits: int, embargo: int) -> list:
    """Build expanding-window train/test splits with an embargo gap.

    Args:
        n_samples (int): total number of chronologically ordered samples.
        n_splits (int): number of test folds.
        embargo (int): samples immediately before each test block to drop from training.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: one (train_idx, test_idx) pair per fold.
    """
    # Your code here
    fold_size = n_samples//(n_splits+1)
    ans = []
    for k in range(1,n_splits+1):
        if k==n_splits:
            test_block = np.arange(k*fold_size, n_samples, 1)
        else:
            test_block = np.arange(k*fold_size, (k+1)*fold_size, 1)
        train_block = np.arange(0, k*fold_size-embargo,1)
        ans.append((train_block, test_block))
    return ans

