import numpy as np
import matplotlib.pyplot as plt


# 阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int)


# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU函数
def ReLU(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# y = sigmoid(x)
y = ReLU(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定Y轴范围
plt.show()
