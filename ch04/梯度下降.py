import numpy as np
from 梯度 import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
        f: 进行优化的函数
        init_x: 初始值
        lr:学习率
        step_num: 指定重复的次数
    """
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


# 使用梯度法求f(x0 + x1) = x0 ** 2 + x1 ** 2的最小值
def function_2(x):
    return x[0] ** 2 + x[1] ** 2


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, 0.1, 100))
