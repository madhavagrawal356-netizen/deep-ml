import numpy as np
def softplus(x: float) -> float:
	"""
	Compute the softplus activation function.

	Args:
		x: Input value

	Returns:
		The softplus value: log(1 + e^x)
	"""

	# Your code here
	if np.exp(x)==np.inf:
		val =  x
	elif 1/np.exp(x)==np.inf:
		val =  0
	else:
		val=np.log(1+np.exp(x))
	return round(val,4)