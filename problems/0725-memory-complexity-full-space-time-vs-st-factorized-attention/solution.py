def attention_memory(T: int, H: int, W: int, num_heads: int, bytes_per_element: int) -> list:
    """
    Compute memory (in bytes) of attention score tensors for full space-time
    attention and for space-time factorized attention.

    Args:
        T: number of temporal frames
        H: spatial height in tokens
        W: spatial width in tokens
        num_heads: number of attention heads
        bytes_per_element: bytes per stored score value (e.g. 4 for fp32, 2 for fp16)

    Returns:
        [full_bytes, factorized_bytes]
    """
    N = T*H*W 
    meme = num_heads*N**2*bytes_per_element 
    S = H*W 
    meme2 = num_heads*(T*S**2+S*T**2)*bytes_per_element
    return [meme, meme2]