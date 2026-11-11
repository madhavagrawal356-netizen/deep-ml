import numpy as np

def forward_kinematics(l1, l2, theta1, theta2):
    # Return (position, jacobian)
    position = (l1*np.cos(theta1)+l2*np.cos(theta1+theta2), l1*np.sin(theta1)+l2*np.sin(theta1+theta2))
    J = [[-l1*np.sin(theta1)-l2*np.sin(theta1+theta2), -l2*np.sin(theta1+theta2)], [l1*np.cos(theta1)+l2*np.cos(theta1+theta2), l2*np.cos(theta1+theta2)]]
    return position, J

