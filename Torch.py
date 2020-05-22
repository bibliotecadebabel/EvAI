import torch

x = [1, 2, 3, 4, 5]
x_2 = x[-10000:]

print(x_2)

x_2.append(10)
print(x_2)
print(x)
'''
x.resize_(3,2,4,4)
print("new x")
print(x.shape)
print(x)
'''