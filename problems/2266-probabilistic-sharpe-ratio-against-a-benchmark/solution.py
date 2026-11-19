import math
import numpy as np


def probabilistic_sharpe(returns, benchmark_sr=0.0, periods_per_year=252):
    """Return the annualized Sharpe, skew, kurtosis, Sharpe standard error and probabilistic Sharpe ratio."""
    returns = np.array(returns)
    if (len(returns)<3 or np.std(returns)==0):
        raise ValueError
    r_bar = np.mean(returns)
    s = np.std(returns, ddof=1)
    sr = r_bar/s 
    annualized_sr = sr*periods_per_year**0.5
    def m_k(returns, r_bar, k):
        return sum((returns-r_bar)**k)/len(returns)
    skewness = m_k(returns,r_bar,3)/(m_k(returns,r_bar,2)**1.5)
    kurtosis = m_k(returns, r_bar,4)/(m_k(returns,r_bar,2)**2)
    sr_se = np.sqrt((1-skewness*sr+(kurtosis-1)*sr**2/4)/(len(returns)-1))
    srr = benchmark_sr/np.sqrt(periods_per_year)
    def norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    psr = norm_cdf((sr-srr)/sr_se)
    return {'sharpe': annualized_sr, 'skew': skewness, 'kurtosis': kurtosis, 'se': sr_se, 'psr':psr}
