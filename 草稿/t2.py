import numpy as np

# 创建一个矩阵w
w = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# 创建一个由100个w组成的矩阵
matrix_100w = np.vstack([w] * 100)
print(matrix_100w)