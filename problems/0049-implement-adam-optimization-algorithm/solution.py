import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    gt=np.zeros(len(x0))
    x=x0
    mt=np.zeros(len(x0))
    vt=np.zeros(len(x0))
    m_cap=np.zeros(len(x0))
    v_cap=np.zeros(len(x0))
    t=0
    for i in range(num_iterations):
        t=i+1
        gt=grad(x)
        for i in range(len(gt)):
            mt[i]=beta1*mt[i]+(1-beta1)*gt[i]
            vt[i]=beta2*vt[i]+(1-beta2)*gt[i]**2
            m_cap[i]=mt[i]/(1-beta1**t)
            v_cap[i]=vt[i]/(1-beta2**t)
            x[i]=x[i]-learning_rate*m_cap[i]/(v_cap[i]**0.5+epsilon)
    return x


