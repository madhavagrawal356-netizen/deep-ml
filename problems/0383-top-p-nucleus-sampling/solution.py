import numpy as np
from scipy.special import softmax
def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    logits = np.array(logits)
    probs = softmax(logits)
    summ = 0
    maxi = np.argsort(-probs, kind='stable')
    liss = set()
    for num in maxi:
        summ+=probs[num]
        liss.add(num)
        if summ>=p:
            break
    for i in range(len(probs)):
        if i not in liss:
            probs[i]=0
    return probs/summ