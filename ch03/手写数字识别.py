import pickle
import numpy as np
from dataset.mnist import load_mnist
from ch03.神经网络信号传递实现 import sigmoid, softmax
from PIL import Image

"""
Downloading train-images-idx3-ubyte.gz ... 
Done
Downloading train-labels-idx1-ubyte.gz ... 
Done
Downloading t10k-images-idx3-ubyte.gz ... 
Done
Downloading t10k-labels-idx1-ubyte.gz ... 
Done
Converting train-images-idx3-ubyte.gz to NumPy Array ...
Done
Converting train-labels-idx1-ubyte.gz to NumPy Array ...
Done
Converting t10k-images-idx3-ubyte.gz to NumPy Array ...
Done
Converting t10k-labels-idx1-ubyte.gz to NumPy Array ...
Done
Creating pickle file ...
Done!
(60000, 784) (60000,) (10000, 784) (10000,)
"""


# 显示MNIST图像
# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()
#
#
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
#
# # print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)
# img, label = x_train[0], t_train[0]
#
# img = img.reshape(28, 28)
# img_show(img)


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + B1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + B2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + B3
    y = softmax(a3)
    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0

# 普通处理
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)  # 获得概率最高的索引
#     if p == t[i]: accuracy_cnt += 1

# 批处理
batch_size = 100  # 批数量
for i in range(0, len(x), batch_size):
    x_batch = x[i: i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i: i+batch_size])

print("准确率：" + str(float(accuracy_cnt) / len(x) * 100) + "%")
