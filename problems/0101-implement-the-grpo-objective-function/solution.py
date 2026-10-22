import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities (per-sample, not normalized).
        pi_theta_ref: List of reference policy probabilities (per-sample, not normalized).
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value.
    """
    # Your code here
    rhos = np.array(rhos)
    pi_theta_old = np.array(pi_theta_old)
    pi_theta_ref = np.array(pi_theta_ref)
    pi_theta = rhos*pi_theta_old
    clipped_rhos = np.clip(rhos, 1-epsilon,1+epsilon)
    surr1 = rhos*A 
    surr2 = clipped_rhos*A 
    policy_objective = np.minimum(surr1,surr2)
    kl_ratio = pi_theta_ref / pi_theta
    kl_penalty = kl_ratio - np.log(kl_ratio) - 1

    # Average objective over samples
    objective = np.mean(
        policy_objective - beta * kl_penalty
    )

    return float(objective)

    