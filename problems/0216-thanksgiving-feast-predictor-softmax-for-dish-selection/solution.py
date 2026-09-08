import numpy as np
def thanksgiving_dish_predictor(preference_scores: list[float]) -> list[float]:
	output=[]
	maxi=max(preference_scores)
	sums=sum(np.exp(t-maxi) for t in preference_scores)
	for item in preference_scores:
		output.append(np.exp(item-maxi)/sums)
	return output



	# Your code here
	pass