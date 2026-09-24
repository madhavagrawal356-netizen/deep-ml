import numpy as np

def adamax_optimizer(parameter, grad, m, u, t, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = beta1*m+(1-beta1)*grad
    u=np.maximum(beta2*u, abs(grad))
    m_cap=m/(1-beta1**t)
    parameter=parameter-learning_rate*m_cap/(u+epsilon)

    return np.round(parameter, 5), np.round(m, 5), np.round(u, 5)