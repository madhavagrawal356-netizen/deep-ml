import numpy as np

def prelu_forward(x: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Implements the forward pass of PReLU.
    Args:
        x: Input array of any shape
        alpha: Slope parameter for negative values (default: 0.25)
    Returns:
        np.ndarray: PReLU activation output, same shape as x
    """
    output=[]
    for item in x:
        if item<0:
            output.append(alpha*item)
        else:
            output.append(item)
    return np.array(output)

    # Your code here
    pass


def prelu_backward(x: np.ndarray, alpha: float, grad_output: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Implements the backward pass of PReLU, computing gradients for both x and alpha.
    Args:
        x: Original input from forward pass
        alpha: Slope parameter used in forward pass
        grad_output: Upstream gradient, same shape as x
    Returns:
        grad_x: Gradient w.r.t. input x, same shape as x
        grad_alpha: Gradient w.r.t. alpha (scalar, summed over all elements)
    """
    dl_dx=[]
    for i in range(len(x)):
        if x[i]>0:
            dl_dx.append(grad_output[i])
        else:
            dl_dx.append(grad_output[i]*alpha)
    grad_alpha=0
    for i in range(len(x)):
        if x[i]<=0:
            grad_alpha+=grad_output[i]*x[i]
    return np.array(dl_dx), grad_alpha
    # Your code here
    pass