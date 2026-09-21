import numpy as np

def qlora_forward(
	x: list[list[float]],
	quantized_W: list[list[int]],
	scale: float,
	zero_point: float,
	A: list[list[float]],
	B: list[list[float]],
	alpha: float = 1.0
) -> list[list[float]]:
	w = np.array(quantized_W)*scale+zero_point
	h = np.array(x)@w + alpha*(np.array(x)@np.array(B)@np.array(A))/(np.array(A).shape[0])
	return list(h)

