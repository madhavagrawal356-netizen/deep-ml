import numpy as np

def softmax(values):
	output=[]
	maxi = max(values)
	main_sum = sum(np.exp(t-maxi) for t in values)
	for t in values:
		f = np.exp(t-maxi)/main_sum
		output.append(f)
	return np.array(output)
	# Implement the softmax function
	pass

def pattern_weaver(n, crystal_values, dimension):
	# Your code here
	attention_scores=[]
	for i in range(len(crystal_values)):
		score=[]
		for j in range(len(crystal_values)):
			score.append(crystal_values[i] * crystal_values[j] / np.sqrt(dimension))
		score=softmax(score)
		attention_scores.append(score)
	x=[]
	crystal_values=np.array(crystal_values)
	#crystal_values=crystal_values.reshape(-1,1)
	for item in attention_scores:
		x.append(item@crystal_values)


	return np.round(x,5)