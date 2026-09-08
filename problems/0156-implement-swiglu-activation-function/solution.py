import numpy as np

def sigmoid(x):
    x = np.array(x)
    return np.array(1/(1+np.exp(-x)))

def SwiGLU(x: np.ndarray) -> np.ndarray:
    scores=[]
    for item in x:
        x1=[]
        x2=[]
        r=int(len(item)/2)
        for i in range(r):
            x1.append(item[i])
            x2.append(item[i+r])
        scr=(np.array(x1))*(np.array(x2))*(sigmoid(x2))
        scores.append(scr)
        
            
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Your code here

    return np.array(scores)
