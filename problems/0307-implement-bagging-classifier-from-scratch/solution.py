import numpy as np

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    n_samples, n_features = X_train.shape
    predictions = []

    for _ in range(n_estimators):

        # 1. Create bootstrap sample
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        best_error = float('inf')
        best_feature = None
        best_threshold = None
        best_direction = None

        # 2. Find the best decision stump
        for feature in range(n_features):

            values = np.unique(X_boot[:, feature])

            # Thresholds between consecutive unique values
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:

                # Direction 1: predict 1 if feature <= threshold
                pred1 = (X_boot[:, feature] <= threshold).astype(int)

                # Direction 2: predict 1 if feature > threshold
                pred2 = (X_boot[:, feature] > threshold).astype(int)

                error1 = np.sum(pred1 != y_boot)
                error2 = np.sum(pred2 != y_boot)

                if error1 < best_error:
                    best_error = error1
                    best_feature = feature
                    best_threshold = threshold
                    best_direction = 1

                if error2 < best_error:
                    best_error = error2
                    best_feature = feature
                    best_threshold = threshold
                    best_direction = 2

        # 3. Predict using the trained stump
        if best_direction == 1:
            pred = (X_test[:, best_feature] <= best_threshold).astype(int)
        else:
            pred = (X_test[:, best_feature] >= best_threshold).astype(int)

        predictions.append(pred)

    # 4. Majority voting
    predictions = np.array(predictions)

    final_predictions = (np.sum(predictions, axis=0) > n_estimators / 2).astype(int)

    return final_predictions