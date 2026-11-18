import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros(b.shape[0])

    for _ in range(n):
        x_new = np.zeros_like(x)

        for i in range(len(b)):
            x_new[i] = (
                b[i] - sum(A[i][j] * x[j]
                           for j in range(len(b)) if j != i)
            ) / A[i][i]

        x = x_new.copy()

    return x