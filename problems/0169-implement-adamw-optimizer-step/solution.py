import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter vector (np.ndarray)
      g: gradient vector (np.ndarray)
      m: first moment vector (np.ndarray)
      v: second moment vector (np.ndarray)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    # Your code here
    m_t = beta1*m+(1-beta1)*g
    v_t = beta2*v+(1-beta2)*g**2
    m_cap = m_t/(1-beta1**t)
    v_cap = v_t/(1-beta2**t)
    w_new = (w-lr*weight_decay*w)-lr*m_cap/(np.sqrt(v_cap)+epsilon)
    return w_new, m_t, v_t