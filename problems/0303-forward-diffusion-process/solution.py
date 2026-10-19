import numpy as np

def forward_diffusion(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray) -> np.ndarray:
    """
    Apply forward diffusion process to add noise to input data.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: Noise array (same shape as x_0)
    
    Returns:
        Noisy sample x_t as numpy array
    """
    beta_schedule = np.linspace(beta_start, beta_end, num_timesteps)
    alpha = 1-beta_schedule
    mul = 1
    alpha_bar=[]
    for item in alpha:
        mul*=item
        alpha_bar.append(mul)
    alpha_bar = np.array(alpha_bar)
    
    a_bar_t = alpha_bar[t - 1]          # t is 1-indexed

    return np.sqrt(a_bar_t) * x_0 + np.sqrt(1 - a_bar_t) * noise


