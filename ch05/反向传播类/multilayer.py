class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 前向传播
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    # 反向传播
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
