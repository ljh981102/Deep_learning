
# 数值微分使用数值方法近似求解函数的导数的过程


def simple_function(x):
    """简单示例函数"""
    return 0.01 * x ** 2 + 0.1 * x


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


print(numerical_diff(simple_function, 5))

# 偏导数仅对某一变量求导数，其他变量是为常数
