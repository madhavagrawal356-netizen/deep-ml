import numpy as np

def overlapping_kv_compression(Z_a, Z_b, B_a, B_b, C_a, C_b):
    """
    Compute compressed KV entries from two overlapping score/value blocks.

    Returns:
        Nested list of shape (N, D).
    """
    Z_a = np.asarray(Z_a, dtype=float)
    Z_b = np.asarray(Z_b, dtype=float)
    B_a = np.asarray(B_a, dtype=float)
    B_b = np.asarray(B_b, dtype=float)
    C_a = np.asarray(C_a, dtype=float)
    C_b = np.asarray(C_b, dtype=float)

    # 1. Add positional biases to the score matrices
    scores_a = Z_a + B_a
    scores_b = Z_b + B_b

    # 2. Concatenate scores for joint attention
    scores = np.concatenate([scores_a, scores_b], axis=1)

    # 3. Numerically stable softmax across both blocks
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)

    # 4. Split attention weights back into the two blocks
    M_a = Z_a.shape[1]
    W_a = weights[:, :M_a]
    W_b = weights[:, M_a:]

    # 5. Compute the jointly compressed output
    C_comp = W_a @ C_a + W_b @ C_b

    return C_comp.tolist()