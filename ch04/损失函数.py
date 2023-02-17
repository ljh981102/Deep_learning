import numpy as np
from dataset.mnist import load_mnist


def mean_squared_error(y, t):
    """均方误差 y->神经网络输出；t->监督数据"""
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    """交叉熵误差 y->神经网络输出；t->监督数据"""
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size, delta = y.shape[0], 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


t = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # 正确解标签对应的索引值是1
y = [0.1, 0.05, 0.0, 0.1, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(mean_squared_error(np.array(y), np.array(t)))
print(cross_entropy_error(np.array(y), np.array(t)))


# 读入MNIST数据集
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10  # 随机抽取10笔数据
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_train = t_train[batch_mask]

print(x_batch, t_train)
