import numpy as np

def amsgrad_step(theta, g, m, v, v_hat, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    theta = np.array(theta)
    g=np.array(g)
    m=np.array(m)
    v=np.array(v)
    v_hat = np.array(v_hat)
    m = beta1*m+(1-beta1)*g 
    v= beta2*v+(1-beta2)*g**2
    v_hat = np.maximum(v,v_hat)
    theta -=lr*m/(np.sqrt(v_hat)+eps)
    return theta,m,v,v_hat

def yogi_step(theta, g, m, s, lr, beta1=0.9, beta2=0.999, eps=1e-3):
    m = beta1*m+(1-beta1)*g 
    s+=(1-beta2)*g**2*np.sign(g**2-s)
    theta-=lr*m/(np.sqrt(s)+eps)
    return theta,m,s

def adopt_step(theta, g, m, s, lr, beta1=0.9, beta2=0.9999, eps=1e-6):
    m=beta1*m+(1-beta1)*g/(np.sqrt(s)+eps)
    theta-=lr*m 
    s = beta2*s+(1-beta2)*g**2 
    return theta,m,s
