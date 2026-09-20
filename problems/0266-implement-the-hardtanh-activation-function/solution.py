def hardtanh(x: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
	"""
	Compute the Hardtanh activation function.

	Args:
		x: Input value
		min_val: Minimum value for the output range (default: -1.0)
		max_val: Maximum value for the output range (default: 1.0)

	Returns:
		The Hardtanh value clipped to [min_val, max_val]
	"""
	if x<min_val:
		return min_val
	elif x>max_val:
		return max_val
	else:
		return x