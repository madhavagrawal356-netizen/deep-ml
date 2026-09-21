import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	x2 = w1@x
	x2 = np.maximum(0, x2)
	x3 = w2@x2
	x= x3+x2
	return np.maximum(0,x)