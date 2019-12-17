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
    network = nw.Network(networkADN, cudaFlag=False)
    network2 = nw.Network(((0, 3, ks[0], x, y),
                            (1, ks[0], 2),
                            (2,)),
                           cudaFlag=False)

    for _,a in enumerate(batch):
        for i in range(1, 80):
            network.Training(data=a[0], p=200, dt=0.01, labels=a[1])
            network2.Training(data=a[0], p=200, dt=0.01, labels=a[1])
            print("Original Network: ", network.total_value," (Filtros=", network.adn[0][2],")")
            print("Mutated Network: ", network2.total_value, " (Filtros=", network2.adn[0][2],")")
            network2 = network2.addFilters()
            network = network.clone()


def Test_LongNetwork(dataGen):

    batch = [dataGen.data]
    print("len data: ", len(dataGen.data[0]))
    ks = [50]
    print("image size: ",dataGen.size[1],"x",dataGen.size[2])
    kernel_1 = 2
    kernel_2 = 2
    kernel_3 = 2
    
    networkADN = ((0, 3, ks[0], kernel_1, kernel_1), (0,ks[0], ks[0]//2, kernel_2,kernel_2), (0,ks[0]//2, ks[0]//4, kernel_3,kernel_3), (1, ks[0]//4, 10), (1, 10, 2), (2,))
    network = nw.Network(networkADN, cudaFlag=False)

    for _,a in enumerate(batch):
        for i in range(1, 80):
            network.Training(data=a[0], p=200, dt=0.01, labels=a[1])
            print("Energy: ", network.total_value)

dataGen = GeneratorFromImage.GeneratorFromImage(2, 100, cuda=False)
dataGen.dataConv2d()
size = dataGen.size



#x = size[1]
#y = size[2]
#k = 2


Test_LongNetwork(dataGen)
