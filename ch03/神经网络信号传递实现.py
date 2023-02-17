import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    """恒等函数"""
    return x


def softmax(a):
    """softmax函数"""
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# 3层神经网络实现

## 输入层向第一层信号传递
X = np.array([1, 0.5])  # 1行2列
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 2行3列
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1  # 1行3列
print(A1)
Z1 = sigmoid(A1)
print(Z1)

## 第一层向第二层信号传递
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
print(A2)
Z2 = sigmoid(A2)
print(Z2)


## 第二层到输出层的信号传递

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
print(A3)
Y = identity_function(A3)
print(Y)
