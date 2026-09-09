import numpy as np
def swish(x: float) -> float:
	"""
	Implements the Swish activation function.

	Args:
		x: Input value

	Returns:
		The Swish activation value
	"""
	return x*(1/(1+np.exp(-x)))
	# Your code here
	pass