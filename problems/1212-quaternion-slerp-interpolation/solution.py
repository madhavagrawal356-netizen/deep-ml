import numpy as np

def slerp(q0, q1, t):
    # q0, q1: lists of 4 numbers [w, x, y, z]
    # t: float in [0, 1]
    # return: list of 4 floats
    q0,q1 = np.array(q0), np.array(q1)

    if q0@q1<0:
        q1=-q1
    theta = np.arccos(q0@q1)

    if np.cos(theta)>0.995:
        return (q0+t*(q1-q0)).tolist() 
    return (np.sin((1-t)*theta)*q0/np.sin(theta)+np.sin(t*theta)*q1/np.sin(theta)).tolist()
