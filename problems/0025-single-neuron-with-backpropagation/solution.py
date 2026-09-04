import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	def forward_pass(features, weights, bias):
		probabilities=[]
		for i in range(len(labels)):
			probabilities.append(1/(1+np.exp(-(sum(x1*x2 for x1,x2 in zip(features[i], weights))+bias))))
		return probabilities
	weights, bias=initial_weights, initial_bias
	mse_values = []
	for i in range(epochs):
		probab=forward_pass(features, weights, bias)
		mse_values.append(sum((y1-y2)**2 for y1,y2 in zip(probab, labels))/len(labels))

		for j in range(len(weights)):
			dl_dwj=(sum((y1-y2)*y1*(1-y1)*x[j] for y1,y2,x in zip(probab, labels, features)))*2/len(labels)
			weights[j]-=learning_rate*dl_dwj
		dl_db= (sum((y1-y2)*y1*(1-y1) for y1,y2 in zip(probab, labels)))*2/len(labels)
		bias -= learning_rate*dl_db 
	updated_weights=weights
	updated_bias=bias


	return updated_weights, updated_bias, mse_values