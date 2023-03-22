"""
随机梯度下降法，更新参数
"""


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr  # 学习率

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
