class AddLayer:
    def __init__(self):
        ...

    # 前向传播
    def forward(self, x, y):
        return x + y

    # 反向传播
    def backword(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
