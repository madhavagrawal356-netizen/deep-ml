
import numpy as np

def reinforce_baseline_update(
    episode: list,
    theta: np.ndarray,
    w: np.ndarray,
    gamma: float,
    alpha_theta: float,
    alpha_w: float
) -> dict:

    theta = np.array(theta, dtype=float).copy()
    w = np.array(w, dtype=float).copy()

    T = len(episode)
    returns = np.zeros(T)
    advantages = np.zeros(T)

    # Step 1: Calculate discounted returns
    G = 0.0
    for t in reversed(range(T)):
        state, action, reward = episode[t]
        G = reward + gamma * G
        returns[t] = G

    # Step 2: Sequential parameter updates
    for t, (state, action, reward) in enumerate(episode):

        x = np.asarray(state, dtype=float)

        # Current baseline prediction
        baseline = w @ x

        # Advantage
        delta = returns[t] - baseline
        advantages[t] = delta

        # Update baseline parameters
        w += alpha_w * delta * x

        # Current policy probabilities
        logits = x @ theta
        logits -= np.max(logits)

        probs = np.exp(logits)
        probs /= np.sum(probs)

        # Gradient of log policy
        grad_log_pi = -np.outer(x, probs)
        grad_log_pi[:, action] += x

        # Update policy parameters
        theta += alpha_theta * (gamma ** t) * delta * grad_log_pi

    return {
        'theta': theta.tolist(),
        'w': w.tolist(),
        'returns': returns.tolist(),
        'advantages': advantages.tolist()
    }
