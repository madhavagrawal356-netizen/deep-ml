import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	Wx=np.array(Wx)
	b=np.array(b)
	initial_hidden_state=np.array(initial_hidden_state)
	input_sequence=np.array(input_sequence)
	Wh=np.array(Wh)
	# Your code here
	final_hidden_state=initial_hidden_state
	for i in range(len(input_sequence)):
		final_hidden_state=np.tanh(Wx@input_sequence[i]+Wh@final_hidden_state+b)
	return np.round(final_hidden_state, decimals=4)