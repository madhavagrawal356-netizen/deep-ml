import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))
	
	def sigmoid(self, func_value):
		return 1/(1+np.exp(-func_value))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		initial_hidden_state=initial_hidden_state
		ct=initial_cell_state
		t=0
		for i in range(len(x)):
			xt = x[i].reshape(-1, 1)
			t=i+1
			combined=np.vstack((initial_hidden_state,xt))
			ft = self.sigmoid(self.Wf@combined+self.bf)
			it = self.sigmoid(self.Wi@combined+self.bi)
			ct_cap=np.tanh(self.Wc@combined+self.bc)
			ct=ft*ct+it*ct_cap
			ot = self.sigmoid(self.Wo@combined+self.bo)
			initial_hidden_state=ot*np.tanh(ct)
		return ot , initial_hidden_state, ct
