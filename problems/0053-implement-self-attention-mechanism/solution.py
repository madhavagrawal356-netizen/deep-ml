import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def softmax(x):
    output=[]
    maxi = max(x)
    sums = sum(np.exp(t-maxi) for t in x)
    for item in x:
        output.append(np.exp(item-maxi)/sums)
    return np.array(output)

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    _, d = K.shape
    scores = Q@K.T/np.sqrt(d)
    for i in range(len(scores)):
        scores[i]=softmax(scores[i])
    output = scores@V
    return output

    # Your code here
    pass
