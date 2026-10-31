import numpy as np


def exact_step(A, b, theta, d):
    """
    Find the exact step size minimizing L(theta + eta*d)
    for L(theta) = 0.5*theta.T@A@theta + b.T@theta + c.
    """
    numerator = -(d @ (A @ theta + b))
    denominator = d @ (A @ d)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def armijo_backtrack(
    f, grad, theta, d,
    eta0=1.0, c=1e-4, rho=0.5, max_halvings=60
):
    """
    Backtracking line search using the Armijo condition.
    """
    eta = eta0

    f_theta = f(theta)
    grad_theta = grad(theta)
    directional_derivative = d @ grad_theta

    for _ in range(max_halvings):
        candidate = theta + eta * d

        if f(candidate) <= f_theta + c * eta * directional_derivative:
            return float(eta)

        eta *= rho

    # Return the last candidate tried
    return float(eta / rho)


def steepest_descent(
    f, grad, theta0, n_iter,
    rule="armijo", eta=0.1, A=None, b=None
):
    """
    Run steepest descent using fixed, exact, or Armijo step sizes.

    Returns:
        Array of shape (n_iter + 1, D) containing all iterates.
    """
    theta = np.asarray(theta0, dtype=float).copy()
    iterates = [theta.copy()]

    for _ in range(n_iter):

        # Steepest descent direction
        d = -np.asarray(grad(theta), dtype=float)

        # Select step size
        if rule == "fixed":
            eta_t = eta

        elif rule == "exact":
            if A is None or b is None:
                raise ValueError("A and b must be provided for exact line search.")

            eta_t = exact_step(A, b, theta, d)

        elif rule == "armijo":
            eta_t = armijo_backtrack(f, grad, theta, d)

        else:
            raise ValueError("rule must be 'fixed', 'exact', or 'armijo'.")

        # Update parameters
        theta = theta + eta_t * d

        iterates.append(theta.copy())

    return np.array(iterates, dtype=float)
