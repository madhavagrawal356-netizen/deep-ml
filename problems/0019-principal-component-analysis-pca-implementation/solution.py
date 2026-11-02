import numpy as np
def pca(data: np.ndarray, k: int) -> np.ndarray:
    pcas = []

    # Standardize each feature
    X = data - np.mean(data, axis=0)
    X = X / np.std(data, axis=0, ddof=1)

    # Compute covariance matrix
    cov = np.cov(X, rowvar=False)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]

    for i in range(k):
        eigenvector = eigenvectors[:, idx[i]].copy()

        # Apply sign convention
        for val in eigenvector:
            if abs(val) > 1e-10:
                if val < 0:
                    eigenvector *= -1
                break

        pcas.append(eigenvector)

    return np.round(np.array(pcas).T, 4)

    
