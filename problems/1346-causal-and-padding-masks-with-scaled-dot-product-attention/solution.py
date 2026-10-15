import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, key_padding_mask=None, causal=False):
    """
    q: (B, H, T_q, D)
    k: (B, H, T_k, D)
    v: (B, H, T_k, D)
    key_padding_mask: (B, T_k) bool, True = real token (may be attended to),
                      False = padding
    causal: if True, query i may only attend to keys at or before its position
    """
    T_q, T_k = q.size(-2), k.size(-2)
    attn_mask = None

    # causal lower-triangular mask: (T_q, T_k) -> (1, 1, T_q, T_k)
    # diagonal=T_k - T_q aligns the mask to the bottom-right, so it stays
    # correct when T_q != T_k (e.g. decoding with a KV cache)
    if causal:
        causal_mask = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).tril(diagonal=T_k - T_q)
        attn_mask = causal_mask[None, None, :, :]

    # key padding mask: (B, T_k) -> (B, 1, 1, T_k)
    if key_padding_mask is not None:
        pad_mask = key_padding_mask.to(torch.bool)[:, None, None, :]
        # combine: a position is attendable only if both masks allow it
        attn_mask = pad_mask if attn_mask is None else (attn_mask & pad_mask)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
