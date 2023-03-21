"""
误差反向传播法快但可能不准确，可以利用数值微分法验证误差，称为「梯度确认」
"""
import numpy as np

from ch05.误差反向传播法实现神经网络学习 import TwoLayerNet
from dataset.mnist import load_mnist

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

"""
OutPut：
W1:4.091015919052476e-10
b1:2.4907611937308347e-09
W2:5.002356286657556e-09
b2:1.3995138555544794e-07
"""
