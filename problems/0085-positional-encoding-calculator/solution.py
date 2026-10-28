import numpy as np

def pos_encoding(position: int, d_model: int):
	if (position==0 or d_model<=0):
		return -1
	pos_encoding = []
	for pos in range(position):
		arr=[]
		for i in range(d_model):
			if i%2==0:
				arr.append(np.sin(pos/(10000**(2*(i//2)/d_model))))
			else:
				arr.append(np.cos(pos/(10000**(2*(i//2)/d_model))))
		pos_encoding.append(np.array(arr))
	pos_encoding = np.array(pos_encoding)
	pos_encoding = np.float16(pos_encoding)
	return pos_encoding