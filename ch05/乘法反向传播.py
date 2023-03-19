from ch05.反向传播类.multilayer import MulLayer

apple_num = 2
apple = 100
tax = 1.1

# Layer
mul_apple_layer = MulLayer()
mul_tex_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tex_layer.forward(apple_price, tax)
print(price)

# backward
dprice = 1
dapple_price, dtax = mul_tex_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)
