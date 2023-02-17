import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 绘制Sin Cos曲线
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle='--', label="cos")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sin&Cos')
plt.legend()
plt.show()

# 绘制图像
img = imread("../dataset/lena.png")
plt.imshow(img)
plt.show()