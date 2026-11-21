import numpy as np

def stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        base_classifiers: list, meta_classifier, n_folds: int = 5) -> np.ndarray:
    """
    Implement a stacking classifier ensemble.
    """
    n_samples = X_train.shape[0]
    n_base = len(base_classifiers)

    meta_train = np.zeros((n_samples, n_base))
    meta_test = np.zeros((X_test.shape[0], n_base))

    fold_size = n_samples // n_folds

    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n_samples

        # Validation indices
        val_idx = np.arange(start, end)

        # Training indices: all rows outside the validation fold
        train_idx = np.concatenate((
            np.arange(0, start),
            np.arange(end, n_samples)
        ))

        X_fit = X_train[train_idx]
        y_fit = y_train[train_idx]
        X_val = X_train[val_idx]

        for j, clf in enumerate(base_classifiers):
            meta_train[val_idx, j] = clf(X_fit, y_fit, X_val)

    # Generate test meta-features using classifiers trained on all training data
    for j, clf in enumerate(base_classifiers):
        meta_test[:, j] = clf(X_train, y_train, X_test)

    # Train meta-classifier and predict
    predictions = meta_classifier(meta_train, y_train, meta_test)

    return np.asarray(predictions, dtype=int)
        