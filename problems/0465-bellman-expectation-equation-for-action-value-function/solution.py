import numpy as np

def bellman_q_value(states: list, actions: list, transition_probs: dict, rewards: dict, policy: dict, gamma: float, theta: float = 1e-6) -> dict:

    # Initialize Q(s,a) = 0 for all state-action pairs
    Q = {(s, a): 0.0 for s in states for a in actions}

    while True:
        delta = 0.0

        # Update Q-values for every state-action pair
        for s in states:
            for a in actions:

                old_value = Q[(s, a)]
                new_value = 0.0

                # Sum over all possible next states
                for next_s in states:

                    prob = transition_probs.get((s, a, next_s), 0.0)
                    reward = rewards.get((s, a, next_s), 0.0)

                    # Expected reward + discounted expected future value
                    next_value = sum(
                        policy.get((next_s, next_a), 0.0) * Q[(next_s, next_a)]
                        for next_a in actions
                    )

                    new_value += prob * (reward + gamma * next_value)

                Q[(s, a)] = new_value

                delta = max(delta, abs(old_value - new_value))

        # Stop when Q-values converge
        if delta < theta:
            break

    return {key: round(value, 4) for key, value in Q.items()}