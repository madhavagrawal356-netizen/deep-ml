import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    current_predictions = np.asarray(current_predictions, dtype=float)

    # Step 1: Calculate residuals
    residuals = y - current_predictions

    n_samples, n_features = X.shape

    best_mse = float("inf")
    best_predictions = np.full(n_samples, np.mean(residuals))

    # Step 2: Search all features and candidate thresholds
    for j in range(n_features):
        unique_values = np.unique(X[:, j])

        # No valid split if there is only one unique value
        if len(unique_values) < 2:
            continue

        thresholds = (
            unique_values[:-1] + unique_values[1:]
        ) / 2

        for threshold in thresholds:
            left_mask = X[:, j] <= threshold
            right_mask = ~left_mask

            # Both partitions must contain samples
            if not np.any(left_mask) or not np.any(right_mask):
                continue

            left_mean = np.mean(residuals[left_mask])
            right_mean = np.mean(residuals[right_mask])

            stump_predictions = np.where(
                left_mask, left_mean, right_mean
            )

            mse = np.mean(
                (residuals - stump_predictions) ** 2
            )

            if mse < best_mse:
                best_mse = mse
                best_predictions = stump_predictions.copy()

    # Step 3: Update the ensemble predictions
    updated_predictions = (
        current_predictions + learning_rate * best_predictions
    )

    return np.round(updated_predictions, 4).tolist()
