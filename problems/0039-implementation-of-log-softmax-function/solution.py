import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	output=[]
	calculated=0
	sums=0
	max_score=max(scores)
	sums = sum(np.exp(scores[t]-max_score) for t in range(len(scores)))
	for i in range(len(scores)):
		calculated=np.exp(scores[i]-max_score)/sums
		output.append(np.log(calculated))
	return output
    
	pass