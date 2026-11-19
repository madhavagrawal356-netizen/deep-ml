import numpy as np

def random_sample_greedy_search(
    q_table: np.ndarray,
    states: list,
    sample_size: int,
    epsilon: float,
    seed: int
) -> dict:
    """
    Perform Random Sample Greedy action selection for a batch of states.
    """
    rng = np.random.RandomState(seed)

    num_states, num_actions = q_table.shape
    k = min(sample_size, num_actions)

    selected_actions = []
    selected_q_values = []
    greedy_matches = 0

    for state in states:
        q_values = q_table[state]

        # True greedy action (ties broken by smallest index)
        true_greedy = int(np.argmax(q_values))

        # Decide between exploration and exploitation
        if rng.random() < epsilon:
            # Exploration: uniformly random action
            action = int(rng.randint(0, num_actions))

        else:
            # Exploitation: sample k actions without replacement
            sampled_actions = rng.choice(
                num_actions,
                size=k,
                replace=False
            )

            # Sort to ensure ties are broken by smallest index
            sampled_actions = np.sort(sampled_actions)

            # Select action with highest Q-value
            action = int(
                sampled_actions[
                    np.argmax(q_values[sampled_actions])
                ]
            )

        selected_actions.append(action)
        selected_q_values.append(round(float(q_values[action]), 4))

        if action == true_greedy:
            greedy_matches += 1

    match_rate = (
        greedy_matches / len(states) if len(states) > 0 else 0.0
    )

    return {
        "actions": selected_actions,
        "q_values": selected_q_values,
        "greedy_match_rate": round(match_rate, 4)
    }