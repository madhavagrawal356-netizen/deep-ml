import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    # Your code here
    W_q = weights['W_q']
    W_k = weights['W_k']
    W_v = weights['W_v']
    W_o = weights['W_o']
    W1,b1 = weights['W1'], weights['b1']
    W2,b2 = weights['W2'], weights['b2']
    gamma1, beta1 = weights['gamma1'], weights['beta1']
    gamma2,beta2=weights['gamma2'], weights['beta2']
    B, T, d_model = X.shape
    d_k = d_model // num_heads
    def layer_norm(x, gamma, beta):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def split_heads(t):
        # (B, T, d_model) -> (B, H, T, d_k)
        return t.reshape(B, T, num_heads, d_k).transpose(0, 2, 1, 3)

    # multi-head self-attention
    Q = split_heads(X @ W_q)
    K = split_heads(X @ W_k)
    V = split_heads(X @ W_v)

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)      # (B, H, T, T)
    scores = scores - scores.max(axis=-1, keepdims=True)     # numerical stability
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)           # softmax over keys

    heads = attn @ V                                          # (B, H, T, d_k)
    concat = heads.transpose(0, 2, 1, 3).reshape(B, T, d_model)
    attn_out = concat @ W_o

    # residual + norm, then feed-forward, residual + norm
    x1 = layer_norm(X + attn_out, gamma1, beta1)
    ffn = np.maximum(0, x1 @ W1 + b1) @ W2 + b2
    return layer_norm(x1 + ffn, gamma2, beta2)
