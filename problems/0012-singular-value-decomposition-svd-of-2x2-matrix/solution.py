import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    B = A.T @ A
    a = B[0, 0]
    b = B[0, 1]
    d = B[1, 1]

    # Step 2: Compute one Jacobi rotation
    if b == 0:
        V = np.eye(2)
    else:
        tau = (d - a) / (2 * b)
        sign = 1 if tau >= 0 else -1

        t = sign / (abs(tau) + np.sqrt(1 + tau**2))
        c = 1 / np.sqrt(1 + t**2)
        s = t * c

        V = np.array([
            [c, s],
            [-s, c]
        ])

    # Step 3: Diagonalize B
    D = V.T @ B @ V

    # Step 4: Extract and sort singular values
    eigenvalues = np.maximum(np.diag(D), 0)
    idx = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    S = np.sqrt(eigenvalues)

    # Step 5: Compute U
    U = np.zeros((2, 2))
    tol = 1e-12 * S[0]

    for i in range(2):
        if S[i] > tol:
            U[:, i] = (A @ V[:, i]) / S[i]

    # Handle zero singular values
    if S[0] > tol and S[1] <= tol:
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])
    elif S[0] <= tol:
        U = np.eye(2)

    return U, S, V.T
    