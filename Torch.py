import torch

x = torch.randn(3, 2, 5, 5)
x = x.cuda()
print("x")
print(x)

x = x[:,:,:4,:4]
print("test")
print(x)
'''
x.resize_(3,2,4,4)
print("new x")
print(x.shape)
print(x)
'''