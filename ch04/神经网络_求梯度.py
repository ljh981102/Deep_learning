import numpy as np

from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def perdict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.perdict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


def f(W):
    return net.loss(x, t)


if __name__ == '__main__':
    net = simpleNet()
    print("权重：", net.W)

    x = np.array([0.6, 0.9])
    p = net.perdict(x)
    print(p)
    print(np.argmax(p))
    t = np.array([0, 0, 1]) #正确解标签
    print(net.loss(x, t))

    print(numerical_gradient(f, net.W))
