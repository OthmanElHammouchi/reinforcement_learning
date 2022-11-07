import numpy as np

#1.1
P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
P_t = P
for i in range(10):
	P_t = P_t @ P
	print(P_t, '\n \n')

#1.2
P = np.array([[0, 0.5, 0.5], [1, 0, 0], [1, 0, 0]])
P_t = P
for i in range(10):
	P_t = P_t @ P
	print(P_t, '\n \n')

