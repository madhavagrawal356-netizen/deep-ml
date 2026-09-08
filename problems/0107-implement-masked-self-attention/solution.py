import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(x):
	maxi=max(x)
	output=[]
	sums=sum(np.exp(t-maxi) for t in x)
	for item in x:
		output.append(np.exp(item-maxi)/sums)
	return np.array(output)
	

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	_, d = Q.shape
	k=K.T
	score = Q@k/np.sqrt(d)
	masked_score=score+mask
	for i in range(len(masked_score)):
		masked_score[i]=softmax(masked_score[i])
	return masked_score@V

	# Your code here
	pass