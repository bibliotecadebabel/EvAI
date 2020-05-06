import torch

x = torch.randn(2, 7, 3, 3)

print("x")
print(x.shape)

a = x.narrow_copy(1, 0, 2)
b = x.narrow_copy(1, 4, 3)

y = torch.cat((a, b), dim=1)

print(y.shape)