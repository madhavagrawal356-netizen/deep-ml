import numpy as np

def cramers_rule(A, b):
    # Your code here
    A = np.array(A)
    _,cols = A.shape
    main_det = np.linalg.det(A)
    if main_det==0:
        return -1
    ans=[]
    for col in range(cols):
        new_A = A.copy()
        new_A[:,col]=np.array(b)
        x = np.linalg.det(new_A)/main_det 
        ans.append(x) 
    return ans




