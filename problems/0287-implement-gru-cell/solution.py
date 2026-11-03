import numpy as np

def gru_cell(x: np.ndarray, h_prev: np.ndarray,
             W_z: np.ndarray, U_z: np.ndarray, b_z: np.ndarray,
             W_r: np.ndarray, U_r: np.ndarray, b_r: np.ndarray,
             W_h: np.ndarray, U_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Implements a single GRU cell forward pass.
    
    Args:
        x: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        W_z, W_r, W_h: Weight matrices for input
        U_z, U_r, U_h: Weight matrices for hidden state
        b_z, b_r, b_h: Bias vectors
    
    Returns:
        h_next: New hidden state of shape (hidden_size,)
    """
    # Your code here
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    z = sigmoid(W_z@x+U_z@h_prev+b_z)
    r = sigmoid(W_r@x+U_r@h_prev+b_r)
    h_cap = np.tanh(W_h@x+U_h@(r*h_prev)+b_h)
    h_next = (1-z)*h_prev+z*h_cap
    return np.round(h_next,4)