def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Your code here
	if isinstance(y_true, list):
		l = 0
		for i in range(len(y_true)):
			if abs(y_true[i]-y_pred[i])<=delta:
				l+=0.5*(y_true[i]-y_pred[i])**2 
			else:
				l+=delta*(abs(y_true[i]-y_pred[i])-0.5*delta)
		return l/len(y_true)
	else:
		if abs(y_true-y_pred)<=delta:
			return 0.5*(y_true-y_pred)**2
		else:
			return delta*(abs(y_true-y_pred)-0.5*delta)
				
