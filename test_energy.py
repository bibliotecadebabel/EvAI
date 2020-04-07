import children.net2.Network as nw
from utilities import Data_generator
import children.Interfaces as Inter
import decimal

import children.pytorch.Network as nw_p
import children.pytorch.MutateNetwork as MutateNetwork
from DAO import GeneratorFromImage

import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim

decimal.getcontext().prec = 100


def Test_numpy_energy():

    print("### NUMPY TEST ###")
    dataGen = Data_generator.Data_gen()
    dataGen.S=200

    dataGen.gen_data()

    x = dataGen.size[0]
    y = dataGen.size[1]
    k = 3

    network = nw.Network([x,y,k])

    clone = network.clone()
    clone.addFilters()

    energy_2_filters = 0
    energy_3_filters = 1

    i = 0
    stop = False
    while stop == False:
        network.Training(data=dataGen.Data, dt=0.01, p=1)
        clone.Training(data=dataGen.Data, dt=0.01, p=1)
        energy_2_filters = network.total_value
        energy_3_filters = clone.total_value

        i += 1

        if energy_3_filters < energy_2_filters and i > 20:
            stop = True

        if i % 10 == 0:
            print("Iteration #",i)
            print("energy 2 filters network=", energy_2_filters)
            print("energy 3 filters network=", energy_3_filters)
    
    print("last Iteration=", i)
    print("energy 2 filters network=", energy_2_filters)
    print("energy 3 filters network=", energy_3_filters)


def Test_pytorch_energy():

    print("### PYTORCH TEST ###")

    dataGen = GeneratorFromImage.GeneratorFromImage(2, 160, cuda=True)
    dataGen.dataConv2d()
    size = dataGen.size

    x = size[1]
    y = size[2]
    k = 2

    batch = [dataGen.data]

    print("len data: ", len(dataGen.data[0]))
    x = dataGen.size[1]
    y = dataGen.size[2]

    
    networkADN = ((0, 3, 3, x, y), (1, 3, 2), (2,))
    mutationADN = ((0, 3, 3, x-1, y-1), (0, 6, 3, x, y),(1, 3, 2), (2,))
    #mutationADN = ((0, 3, 4, x, y), (1, 4, 2), (2,))
    network = nw_p.Network(networkADN, cudaFlag=True)
    clone = MutateNetwork.executeMutation(network, mutationADN)

    energy_2_filters = 0
    energy_3_filters = 1
    stop = False
    i = 0
    for _,a in enumerate(batch):

        while stop == False:
            #print("entrenando red #1")
            network.Training(data=a[0], p=10, dt=0.01, labels=a[1])
            #print("entrenando red #2")
            clone.Training(data=a[0], p=10, dt=0.01, labels=a[1])
            energy_2_filters = network.total_value
            energy_3_filters = clone.total_value
            i += 1

            if energy_3_filters < energy_2_filters:
                stop = True

            if i % 10 == 0:
                print("Iteration #",i)
                print("energy network 1=", energy_2_filters)
                print("energy network 2=", energy_3_filters)
    
    print("last Iteration=", i)
    print("energy network 1=", energy_2_filters)
    print("energy network 2=", energy_3_filters)



#Test_numpy_energy()
Test_pytorch_energy()