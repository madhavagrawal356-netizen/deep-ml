
import numpy as np
from itertools import combinations

def solve_zero_sum(payoff):
    """Return (value, row_strategy, col_strategy) of the zero-sum game."""
    A = np.asarray(payoff, dtype=float)
    m, n = A.shape
    tol = 1e-8

    row_candidates = []
    col_candidates = []

    # Find candidate optimal row strategies
    for k in range(1, min(m, n) + 1):
        for I in combinations(range(m), k):
            for J in combinations(range(n), k):
                B = A[np.ix_(I, J)]

                # Solve: p.T @ B = v, sum(p) = 1
                M = np.zeros((k + 1, k + 1))
                M[:k, :k] = B.T
                M[:k, k] = -1
                M[k, :k] = 1

                b = np.zeros(k + 1)
                b[k] = 1

                try:
                    sol = np.linalg.solve(M, b)
                except np.linalg.LinAlgError:
                    continue

                p_support = sol[:k]
                v = sol[k]

                if np.any(p_support < -tol):
                    continue

                p = np.zeros(m)
                p[list(I)] = p_support

                # Row strategy must guarantee at least v
                if np.min(p @ A) >= v - tol:
                    row_candidates.append((v, p))

    # Find candidate optimal column strategies
    for k in range(1, min(m, n) + 1):
        for I in combinations(range(m), k):
            for J in combinations(range(n), k):
                B = A[np.ix_(I, J)]

                # Solve: B @ q = v, sum(q) = 1
                M = np.zeros((k + 1, k + 1))
                M[:k, :k] = B
                M[:k, k] = -1
                M[k, :k] = 1

                b = np.zeros(k + 1)
                b[k] = 1

                try:
                    sol = np.linalg.solve(M, b)
                except np.linalg.LinAlgError:
                    continue

                q_support = sol[:k]
                v = sol[k]

                if np.any(q_support < -tol):
                    continue

                q = np.zeros(n)
                q[list(J)] = q_support

                # Column strategy must hold the payoff to at most v
                if np.max(A @ q) <= v + tol:
                    col_candidates.append((v, q))

    # Match row and column optimal strategies
    for vr, p in row_candidates:
        for vc, q in col_candidates:
            if abs(vr - vc) <= tol:
                p = np.maximum(p, 0)
                q = np.maximum(q, 0)

                p /= np.sum(p)
                q /= np.sum(q)

                return (
                    float((vr + vc) / 2),
                    p.tolist(),
                    q.tolist()
                )

    raise ValueError("Could not find an equilibrium.")

