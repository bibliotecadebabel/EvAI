import children.pytorch.Network as nw

from DAO import GeneratorFromImage

import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim

def Test_noCuda(dataGen):
    batch = [dataGen.data]
    print("len data: ", len(dataGen.data[0]))
    ks = [50]
    x = dataGen.size[1]
    y = dataGen.size[2]

    print("creating networks")
    #(0, ks[i], len(dataGen.data[0]), 1, 1),
    networkADN = ((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
    objects = [x, y, ks[0]]
    network = nw.Network(networkADN, objects, cudaFlag=False)
    network2 = nw.Network(((0, 3, ks[0], x, y), (1, ks[0], 2), (2,)), [x, y, ks[0]], cudaFlag=False)

    for _,a in enumerate(batch):
        for i in range(1, 80):
            network.Training(data=a[0], p=200, dt=0.01, labels=a[1])
            network2.Training(data=a[0], p=200, dt=0.01, labels=a[1])
            print("Original Network: ", network.total_value," (Filtros=", network.objects[2],")")
            print("Mutated Network: ", network2.total_value, " (Filtros=", network2.objects[2],")")
            network2 = network2.addFilters()
            network = network.clone()

dataGen = GeneratorFromImage.GeneratorFromImage(2, 100, cuda=False)
dataGen.dataConv2d()
size = dataGen.size


x = size[1]
y = size[2]
k = 2


Test_noCuda(dataGen)