import numpy as np

def irope_attention(Q: list, K: list, V: list, positions: list,
                    layer_index: int, rope_layers: list,
                    base: float = 10000.0) -> dict:

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    Q = np.array(Q, dtype=float)
    K = np.array(K, dtype=float)
    V = np.array(V, dtype=float)

    seq_len, d_head = K.shape

    if layer_index not in rope_layers:
        scores = Q @ K.T / np.sqrt(d_head)
        attention_weights = softmax(scores)
        output = attention_weights @ V

        return {
            'output': np.round(output, 4).tolist(),
            'attention_weights': np.round(attention_weights, 4).tolist(),
            'uses_rope': False
        }

    else:
        positions = np.array(positions)
        i = np.arange(0, d_head, 2)

        inv_freq = 1.0 / (base ** (i / d_head))
        angles = np.outer(positions, inv_freq)

        rotation = np.exp(1j * angles)

        Q_complex = Q[:, 0::2] + 1j * Q[:, 1::2]
        K_complex = K[:, 0::2] + 1j * K[:, 1::2]

        Q_rot = Q_complex * rotation
        K_rot = K_complex * rotation

        Q[:, 0::2] = Q_rot.real
        Q[:, 1::2] = Q_rot.imag

        K[:, 0::2] = K_rot.real
        K[:, 1::2] = K_rot.imag

        scores = Q @ K.T / np.sqrt(d_head)
        attention_weights = softmax(scores)
        output = attention_weights @ V

        return {
            'output': np.round(output, 4).tolist(),
            'attention_weights': np.round(attention_weights, 4).tolist(),
            'uses_rope': True
        }




        

    