import numpy as np


def function_2(x):
    return x[0]**2 + x[1] ** 2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx, tmp_val in enumerate(x):
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


# 分别求（3，4）（0，2）（3，0）点的梯度
if __name__ == '__main__':
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    print(numerical_gradient(function_2, np.array([3.0, 0.0])))
