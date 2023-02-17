import numpy as np


# 与门
# 与门、与非门、或门是具有相同构造的感知机
def AND(x1, x2):
    """与门 逻辑电路"""
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    return 0 if tmp <= theta else 1


print(AND(0, 0), AND(0, 1), AND(1, 0), AND(1, 1))

x = np.array([0, 1])  # 输入
w = np.array([0.5, 0.5])  # 权重
b = -0.7  # 偏置
print(np.sum(x * w) + b)


# 使用偏置改造上述门电路
def _AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    # w = np.array([0.5, 0.5])
    # b = -0.7
    # 这样是与非门，仅权重、偏置是其负数
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1


print(OR(0, 0), OR(0, 1), OR(1, 0), OR(1, 1))


# 通过组合与、与非、或门实现异或门电路
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return _AND(s1, s2)


print(XOR(0, 0), XOR(0, 1), XOR(1, 0), XOR(1, 1))
