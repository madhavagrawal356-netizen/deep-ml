import math

def sigmoid(z: float) -> float:
	#Your code here
	result = 1/(1+math.exp(-z))
	result = round(result, 4)
	return result