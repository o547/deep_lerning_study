import numpy as np

x = np.array([1.0, 2.0, 3.0])
# print(x)

A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 1], [2, 2]])
# print(A * B)
# print(A[A > 2])

A[0][0]=10
print(A)