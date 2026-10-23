import numpy as np

def bellman_update(V, transitions, gamma):
    new_V = np.zeros_like(V, dtype=float)
    for s in range(len(V)):
        action_values=[]
        for a in transitions[s]:
            value=0.0
            for prob, next_state, reward, done in transitions[s][a]:
                future_value = 0.0 if done else V[next_state]
                value+=prob*(reward+gamma*future_value)
            action_values.append(value)
        new_V[s]=max(action_values)
    return new_V           
  