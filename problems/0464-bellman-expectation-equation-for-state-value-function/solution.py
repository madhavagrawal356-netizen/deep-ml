import numpy as np

def bellman_expectation_value(P, R, policy, gamma):
    """
    Compute the state-value function V^pi for a given policy using
    the Bellman expectation equation.
    
    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states)
        R: Rewards, shape (num_states, num_actions, num_states)
        policy: Stochastic policy, shape (num_states, num_actions)
        gamma: Discount factor
    
    Returns:
        State-value function as numpy array of shape (num_states,)
    """
    num_states, num_actions, _ = P.shape

    # Policy-weighted transition matrix and expected reward vector
    P_pi = np.zeros((num_states, num_states))
    R_pi = np.zeros(num_states)

    for s in range(num_states):
        for a in range(num_actions):
            P_pi[s] += policy[s, a] * P[s, a]
            R_pi[s] += policy[s, a] * np.sum(P[s, a] * R[s, a])

    # Solve V = R_pi + gamma * P_pi @ V
    V = np.linalg.solve(
        np.eye(num_states) - gamma * P_pi,
        R_pi
    )

    return V
    


