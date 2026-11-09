
import numpy as np

def bfgs_inverse_update(C, s, y):
    rho = 1.0 / (y.T @ s)
    I = np.eye(len(s))

    A = I - rho * np.outer(s, y)
    B = I - rho * np.outer(y, s)

    C_new = A @ C @ B + rho * np.outer(s, s)

    return C_new


def lbfgs_direction(g, s_hist, y_hist):
    if len(s_hist) == 0:
        return -g

    q = g.copy()
    M = len(s_hist)
    alpha = np.zeros(M)
    rho = np.zeros(M)

    # First loop: newest to oldest
    for i in range(M - 1, -1, -1):
        s = s_hist[i]
        y = y_hist[i]

        rho[i] = 1.0 / (y @ s)
        alpha[i] = rho[i] * (s @ q)

        q = q - alpha[i] * y

    # Initial inverse-Hessian scaling
    s = s_hist[-1]
    y = y_hist[-1]

    gamma = (s @ y) / (y @ y)
    r = gamma * q

    # Second loop: oldest to newest
    for i in range(M):
        s = s_hist[i]
        y = y_hist[i]

        beta = rho[i] * (y @ r)
        r = r + s * (alpha[i] - beta)

    return -r


def wolfe_line_search(f, grad, x, d, c1=1e-4, c2=0.9, max_iter=60):
    phi0 = f(x)
    phi0_prime = d @ grad(x)

    lo = 0.0
    hi = np.inf
    eta = 1.0

    for _ in range(max_iter):
        x_new = x + eta * d
        phi = f(x_new)

        if phi > phi0 + c1 * eta * phi0_prime:
            hi = eta
            eta = (lo + hi) / 2.0

        elif d @ grad(x_new) < c2 * phi0_prime:
            lo = eta

            if np.isinf(hi):
                eta = 2.0 * lo
            else:
                eta = (lo + hi) / 2.0

        else:
            break

    return eta


def minimize_lbfgs(f, grad, x0, m=5, max_iter=200, tol=1e-8):
    x = np.asarray(x0, dtype=float).copy()

    s_hist = []
    y_hist = []

    n_iter = 0

    for k in range(max_iter):
        g = grad(x)

        if np.linalg.norm(g) < tol:
            break

        d = lbfgs_direction(g, s_hist, y_hist)

        eta = wolfe_line_search(f, grad, x, d,c2=0.5)

        x_new = x + eta * d
        g_new = grad(x_new)

        s = x_new - x
        y = g_new - g

        if y @ s > 1e-12:
            s_hist.append(s.copy())
            y_hist.append(y.copy())

            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)

        x = x_new
        n_iter += 1

        # Check convergence at the new point
        if np.linalg.norm(g_new) < tol:
            break

    return x, n_iter

