import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	probabilities=[]
	for i in range(len(labels)):
		probabilities.append(1/(1+math.exp(-(sum(x1*x2 for x1,x2 in zip(features[i], weights))+bias))))
	mse= (sum((y1-y2)**2 for y1,y2 in zip(probabilities, labels)))/len(labels)
	return probabilities, mse