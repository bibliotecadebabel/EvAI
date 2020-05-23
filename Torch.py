import torch


for i in range(200):
    evens = list(range(i*32, (i+1)*32))
    random_sample = torch.utils.data.SubsetRandomSampler(evens)
    for j in random_sample:
        print(j)