import numpy as np

# 生成numpy数组
x = np.array([1.0, 2.0, 3.0])
print(x)

# 对应元素四则运算
x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])
print(x - y, x + y, x * y, x / y)

# Numpy数组与标量运算，又称(广播)
print(x / 2.0)

# N维数组
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A, A.shape, A.dtype)


# 访问元素
X = np.array([[50, 51], [52, 53], [54, 55]])

# 将X转为一维数组
print(X.flatten())

# 通过np.array指定获取元素
print(X.flatten()[np.array([0, 2, 4])])

# 指定大于52的元素输出
print(X.flatten() > 52, X.flatten()[X.flatten() > 52])
