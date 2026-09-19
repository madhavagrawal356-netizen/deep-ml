import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	sigmoid_act = np.exp(-x)/(1+np.exp(-x))**2
	tanh_act = 1-((1-np.exp(-2*x))/(1+np.exp(-2*x)))**2
	if x>0:
		relu = 1
	else:
		relu=0
	return {'sigmoid': sigmoid_act, 'tanh': tanh_act, 'relu': relu}
