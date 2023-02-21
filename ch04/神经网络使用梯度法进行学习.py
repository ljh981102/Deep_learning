from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
    """参数：输入层神经元数、隐藏层神经元数、输出层神经元数"""

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.prarms = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size),  # 第一层权重
            'b1': np.zeros(hidden_size),  # 第一层偏置
            'W2': weight_init_std * np.random.randn(hidden_size, output_size),  # 第二层权重
            'b2': np.zeros(output_size)  # 第二层偏置
        }

    def predict(self, x):
        """进行推理"""
        W1, W2 = self.prarms['W1'], self.prarms['W2']
        b1, b2 = self.prarms['b1'], self.prarms['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """x：输入数据  t：监督数据"""
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """计算识别精度"""
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """计算权重参数的梯度"""
        loss_W = lambda W: self.loss(x, t)

        # 各层权重偏置的梯度
        grads = {
            'W1': numerical_gradient(loss_W, self.prarms['W1']),
            'b1': numerical_gradient(loss_W, self.prarms['b1']),
            'W2': numerical_gradient(loss_W, self.prarms['W2']),
            'b2': numerical_gradient(loss_W, self.prarms['b2'])
        }

        return grads


# params、grads字典数据示例
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# print(net.prarms['W1'].shape, net.prarms['b1'].shape, net.prarms['W2'].shape, net.prarms['b2'].shape)

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)
grads = net.numerical_gradient(x, t)
print(grads['W1'].shape, grads['b1'].shape, grads['W2'].shape, grads['b2'].shape)
