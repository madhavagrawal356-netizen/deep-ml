import numpy as np
def q_learning(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes):
    Q = np.zeros((num_states, num_actions))
    states = [i for i in range(num_states)]
    for item in terminal_states:
        if item in states:
            states.remove(item)
    for i in range(num_episodes):
        state = np.random.choice(states)
        while state not in terminal_states:
            step = np.random.rand()
            if step<epsilon:
                a = np.random.randint(num_actions)
            else:
                a = np.argmax(Q[state])
            next_state = np.random.choice(num_states, p=P[state,a])
            if next_state in terminal_states:
                target=R[state][a]
            else:
                target=R[state][a]+gamma*np.max(Q[next_state], axis=0)
            Q[state][a]+=alpha*(target-Q[state][a])
            state=next_state
    return Q
        


    