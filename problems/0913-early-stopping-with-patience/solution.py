import numpy as np

class EarlyStopping:
    def __init__(self, patience: int, mode: str = 'min'):
        self.patience = patience
        self.mode = mode
        self.bad_steps = 0

        if mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf

    def step(self, metric: float) -> bool:
        if self.mode == 'min':
            improved = metric < self.best
        else:
            improved = metric > self.best

        if improved:
            self.best = metric
            self.bad_steps = 0
        else:
            self.bad_steps += 1

        return self.bad_steps >= self.patience
        
        
