import numpy as np

def divide_on_feature(X, feature_i, threshold):
    # Your code here
    meets = []
    not_meets=[]
    for item in X:
        if item[feature_i]>=threshold:
            meets.append(item)
        else:
            not_meets.append(item)
    return [np.array(meets), np.array(not_meets)]