def softsign(x: float) -> float:
	"""
	Implements the Softsign activation function.

	Args:
		x (float): Input value

	Returns:
		float: The Softsign of the input
	"""
	result = x/(1+abs(x))
	# Your code here
	return round(result, 4)