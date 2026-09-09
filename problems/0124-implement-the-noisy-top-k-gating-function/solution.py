import numpy as np
def softplus(x):
    return np.log(1+np.exp(x))

def softmax(x):
    output=[]
    maxi = max(x)
    sums=sum(np.exp(t-maxi) for t in x)
    for item in x:
        output.append(np.exp(item-maxi)/sums)
    return np.array(output)
def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    # Your code here
    hbase=X@W_g
    hnoise = X@W_noise
    h = hbase+N*softplus(hnoise)
    h_=np.zeros_like(h)
    for i, item in enumerate(h):
        top_idx = np.argpartition(item, -k)[-k:]
        h_[i, top_idx] = softmax(item[top_idx])
    return h_
    return np.array(h_)
