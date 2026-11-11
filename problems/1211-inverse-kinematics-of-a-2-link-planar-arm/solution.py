import math

def inverse_kinematics(x, y, l1, l2, elbow='up'):
    # x, y: target position
    # l1, l2: link lengths
    # elbow: 'up' or 'down'
    # return [theta1, theta2] in radians, or None if unreachable
    D = (x**2+y**2-l1**2-l2**2)/(2*l1*l2)
    if D>1 or D<-1:
        return None 
    if elbow=='up':
        multi = -1 
    else:
        multi = 1
    theta2 = multi*math.acos(D)   
    theta1 = math.atan2(y,x)-math.atan2(l2*math.sin(theta2), l1+l2*math.cos(theta2))
    return [theta1,theta2]