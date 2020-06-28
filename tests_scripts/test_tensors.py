import torch as torch

oldFilter = torch.zeros(2, 1, 3, 3)
newFilter = torch.ones(2, 2, 3, 3)

mixed = torch.cat((oldFilter, newFilter), dim=1)

print(mixed)