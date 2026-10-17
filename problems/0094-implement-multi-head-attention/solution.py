import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = X@W_q
    K=X@W_k
    V=X@W_v
    return (Q,K,V)
def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = K.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)                              # (seq_len, seq_len)
    scores = scores - np.max(scores, axis=-1, keepdims=True)     # numerical stability
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)  # softmax over keys
    return weights @ V    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = K.shape
    d_k = d_model//n_heads
    heads = []
    for h in range(n_heads):
        sl = slice(h * d_k, (h + 1) * d_k)
        heads.append(self_attention(Q[:, sl], K[:, sl], V[:, sl]))

    return np.concatenate(heads, axis=-1)        

    