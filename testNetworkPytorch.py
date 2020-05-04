import children.pytorch.Network as nw
import children.pytorch.NetworkDendrites as nw_dendrites
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
import children.pytorch.MutateNetwork as MutateNetwork
import children.pytorch.MutateNetwork_Dendrites as Mutate_Dendrites
from DAO import GeneratorFromImage, GeneratorFromCIFAR

from DNA_conditions import max_layer
from DNA_creators import Creator
from DNA_Graph import DNA_Graph

import DNA_directions_f as dire


def DNA_test_i(x,y):
    def condition(DNA):
        output=True
        if DNA:
            for num_layer in range(0,len(DNA)-3):
                layer=DNA[num_layer]
                x_l=layer[3]
                y_l=layer[4]
                output=output and (x_l<x) and (y_l<y)
        if output:
            return max_layer(DNA, 3)
    center=((0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y), (1, 5, 10), (2,))
    version='inclusion'
    space=space=DNA_Graph(center,2,(x,y),condition,(0,(1,0,0,0)),version)
    return space

def remove_layer_test(x,y, indexTarget):
    #center=((-1,x,y),(0, 3, 5, 5, 5),(0, 5, 8, 1,1),(0,13,12, 28, 28),
    #        (1, 12, 10), (2,),(3,-1,0),(3,0,1),
    #        (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    center=((-1,x,y),(0, 3, 2, 5, 5),(0, 2, 2, 1,1),(0,4,2, 7, 7),
            (1, 2, 10), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))

    mutateADN=dire.remove_layer(indexTarget,center)

    return [center, mutateADN]

def add_layer_test(x,y, indexTarget):
    #center=((-1,x,y),(0, 3, 5, 5, 5),(0, 5, 8, 7,7),(0,13,12, 28, 28),
    #        (1, 12, 10), (2,),(3,-1,0),(3,0,1),
    #        (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    center=((-1,x,y),(0, 3, 2, 5, 5),(0, 2, 2, 1,1),(0,4,2, 7, 7),
            (1, 2, 10), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
 
    mutateADN=dire.add_layer(indexTarget,center)

    return [center, mutateADN]

def Test_Mutacion_dendrites():

    indexTarget = 0

    #dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  50)
    dataGen = GeneratorFromImage.GeneratorFromImage(2, 100)
    dataGen.dataConv2d()

    print("creating DNAs")

    dna_list = add_layer_test(dataGen.size[1], dataGen.size[2], indexTarget)

    networkADN = dna_list[0]
    mutate_adn = dna_list[1]   

    network = nw_dendrites.Network(networkADN, cudaFlag=True)
    

    print("adn=", networkADN)
    print("mutateADN=", mutate_adn)

    

    network.Training(data=dataGen, p=100, dt=0.01, labels=None)
    #print("Accuracy original (1)=", network.generateEnergy(dataGen))
    #network.Training(data=dataGen, p=1900, dt=0.01, labels=None)
    #print("Accuracy original (2)=", network.generateEnergy(dataGen))
    network_1 = Mutate_Dendrites.executeMutation(oldNetwork=network, newAdn=mutate_adn)
    network_1.Training(data=dataGen, p=100, dt=0.01, labels=None)
    #print("Accuracy mutate (1)=", network_1.generateEnergy(dataGen))
    #network_1.Training(data=dataGen, p=1900, dt=0.01, labels=None)
    #print("Accuracy mutate (2)=", network_1.generateEnergy(dataGen))

def Test_Mutacion():

    #dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  2)
    dataGen = GeneratorFromImage.GeneratorFromImage(2, 2)
    dataGen.dataConv2d()

    print("creating DNAs")
    
    networkADN = ((0, 3, 1, 2, 2), (0, 4, 2, 11, 11), (1, 2, 2), (2,))
    mutate_adn = ((0, 3, 1, 3, 3), (0, 4, 2, 11, 11), (1, 2, 2), (2,))
    network = nw.Network(networkADN, cudaFlag=True)
    
    network.Training(data=dataGen, p=100, dt=0.01, labels=None)
    #print("Accuracy original=", network.generateEnergy(dataGen))
    network_1 = MutateNetwork.executeMutation(oldNetwork=network, newAdn=mutate_adn)
    network_1.Training(data=dataGen, p=100, dt=0.01, labels=None)
    #print("Accuracy mutate=", network_1.generateEnergy(dataGen))
    

#Test_pytorchNetwork()

#Test_Batch(dataGen)
Test_Mutacion_dendrites()
#Test_Mutacion()
