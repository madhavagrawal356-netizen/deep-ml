import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	X = np.insert(X,0,1,axis=1)
	theta = np.zeros(X.shape[1], dtype=float)	
	res=[]
	for i in range(iterations):
		y_cap =np.array([1/(1+np.exp(-X[i].T@theta)) for i in range(len(X))])
		L = -sum(y[i]*np.log(y_cap[i])+(1-y[i])*np.log(1-y_cap[i]) for i in range(len(y)))
		theta -=learning_rate*X.T@(y_cap-y)
		res.append(L)
	return np.round(theta,4).tolist(),np.round(res,4)



