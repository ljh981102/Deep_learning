import numpy as np
import matplotlib.pyplot as plt

# 一维数组
A = np.array([1, 2, 3, 4])
print(np.ndim(A))  # 数组维数
print(A.shape)  # 数组形状

# 二维数组
B = np.array([[1, 2], [3, 4], [5, 6]])
print(np.ndim(B))
print(B.shape)

# 矩阵乘积
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))
