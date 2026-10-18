import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	ans = []
	f_x = np.array(f(x))
	for j in range(len(x)):
		x_h = list(x)
		x_h[j]+=h
		diff = (np.array(f(x_h)) - f_x) / h
		ans.append(diff.tolist())
	return np.array(ans).T.tolist()

