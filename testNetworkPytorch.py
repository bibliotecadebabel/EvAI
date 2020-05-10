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

def DNA_test_f(x,y):
    def condition(DNA):
        return max_layer(DNA,15)
    center=((-1, 1, 3, 32, 32), (0, 3, 5, 3, 3), (0, 8, 7, 32, 32), (1, 7, 10), (2,), (3, -1, 0), (3, -1, 1), (3, 0, 1), (3, 1, 2), (3, 2, 3))
    version='final'
    space=space=DNA_Graph(center,1,(x,y),condition
        ,((0,0,1),(0,0,1,1),(0,1,0,0),(1,0,0,0)),version)
    
    return space

def remove_layer_test(x,y, indexTarget):
    center=((-1,1,3,x,y),(0, 3, 5, 5, 5),(0, 5, 8, 1,1),(0,13,12, 7, 7),
            (1, 12, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    #center=((-1,1,3,x,y),(0, 3, 2, 5, 5),(0, 2, 2, 1,1),(0,4,2, 7, 7),
    #        (1, 2, 10), (2,),(3,-1,0),(3,0,1),
    #        (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    #center = ((-1,1,3,x,y), (0, 3, 3, 3, 3), (0, 3, 6, 5, 5), (0, 6, 4, 1, 1), (0, 7, 2, 9, 9), (1, 2, 10), (2,), 
    #(3, -1, 0),(3, 0, 3),(3, 0, 1), (3, 1, 2), (3, 2, 3))
    #center=((-1,1,3,x,y),(0, 3, 2, 5, 5),(0, 2, 3, 7,7),(0,5,2, 28, 28),
    #        (1, 2, 10), (2,),(3,-1,0),(3,0,1),
    #        (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    mutateADN=dire.remove_layer(indexTarget,center)

    return [center, mutateADN]

def add_layer_test(x,y, indexTarget):
    #center=((-1,1,3,x,y),(0, 3, 5, 5, 5),(0, 5, 8, 7,7),(0,13,12, 28, 28),
    #        (1, 12, 10), (2,),(3,-1,0),(3,0,1),
    #        (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    center=((-1,1,3,x,y),(0, 3, 2, 5, 5),(0, 2, 2, 1,1),(0,4,2, 7, 7),
            (1, 2, 10), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
 
    mutateADN=dire.add_layer(indexTarget,center)

    return [center, mutateADN]

def increase_filters_test(x,y, indexTarget):
    center= ((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 7, 3,3),(0,15, 12, 11, 11),
            (1, 12, 10), (2,),(3,-1,0),(3,0,1),(3,-1, 2),
            (3,0,2), (3,1,2), (3, 2, 3), (3, 3, 4))
    mutateADN = dire.increase_filters(indexTarget,center)

    return [center, mutateADN]

def decrease_filters_test(x,y, indexTarget):

    center= ((-1,1,3,x,y),(0, 3, 3, 3, 3),(0, 3, 2, 3,3),(0,8,3, 11, 11),
            (1, 3, 2), (2,),(3,-1,0),(3,0,1),
            (3,0,2), (3,1,2), (3,-1, 2), (3, 2, 3), (3, 3, 4))

    mutateADN = dire.decrease_filters(indexTarget,center)

    return [center, mutateADN]

def increase_kernel_test(x,y, indexTarget):

    center= ((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 3,3),(0,13,5, 9, 9),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,0,2), (3, 2, 3), (3, 3, 4))

    mutateADN = dire.increase_kernel(indexTarget,center)

    return [center, mutateADN]

def decrease_kernel_test(x,y, indexTarget):

    center= ((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 3,3),(0,13,5, 9, 9),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,0,2), (3, 2, 3), (3, 3, 4))

    mutateADN = dire.decrease_kernel(indexTarget,center)

    return [center, mutateADN]


def spread_dendrites_test(x,y, indexTarget):
    
    # AGREGA DENDRITA
    #center= ((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 7,7),(0,8,5, 3, 3),
    #        (1, 5, 2), (2,),(3,-1,0),(3,0,1),
    #        (3,1,2),(3,2,3),(3,3,4))

    # AGREGA DENDRITA
    #center = ((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 8, 8, 7,7),(0,13,5, 9, 9),
    #        (1, 5, 2), (2,),(3,-1,0),(3,0,1),(3,-1,1),
    #        (3,1,2),(3,0,2),(3,2,3),(3,3,4))

    # AGREGA DENDRITA
    #center = ((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 7,7),(0,16,5, 11, 11),
    #        (1, 5, 2), (2,),(3,-1,0),(3,0,1),
    #        (3,1,2),(3,0,2),(3,-1,2),(3,2,3),(3,3,4))

    # CAMBIA DENDRITA
    center = ((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 8, 5, 3,3),(0,8,5, 3, 3),
            (0,5,5, 9, 9),(1, 5, 2), (2,),(3,-1,0),(3,0,1),(3,-1,1),
            (3,1,2),(3,-1,2),(3,2,3),(3,3,4),(3,4,5))
            
    mutateADN = dire.spread_dendrites(indexTarget,center)

    return [center, mutateADN]

def retract_dendrites_test(x,y, indexTarget):
    
    center = ((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 8, 8, 7,7),(0,13,5, 9, 9),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),(3,-1,1),
            (3,1,2),(3,0,2),(3,2,3),(3,3,4))
            
    mutateADN = dire.retract_dendrites(indexTarget,center)

    return [center, mutateADN]

def Test_Mutacion_dendrites():

    indexTarget = 0

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  50)
    #dataGen = GeneratorFromImage.GeneratorFromImage(2, 100)
    dataGen.dataConv2d()

    print("creating DNAs")

    dna_list = spread_dendrites_test(dataGen.size[1], dataGen.size[2], indexTarget)

    #networkADN = dna_list[0]
    #mutate_adn = dna_list[1]

    
    networkADN = ((-1, 1, 3, 11, 11), (0, 3, 5, 11, 11), (1, 5, 2), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2))

    mutate_adn = ((-1, 1, 3, 11, 11), (0, 3, 5, 3, 3), (0, 5, 5, 9, 9), (1, 5, 2), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3))
    

    network = nw_dendrites.Network(networkADN, cudaFlag=True)
    

    print("adn=", networkADN)
    print("mutateADN=", mutate_adn)

    

    network.Training(data=dataGen, p=100, dt=0.01, labels=None)
    #print("Accuracy original (1)=", network.generateEnergy(dataGen))
    #network.Training(data=dataGen, p=1900, dt=0.01, labels=None)
    #print("Accuracy original (2)=", network.generateEnergy(dataGen))
    network_1 = Mutate_Dendrites.executeMutation(oldNetwork=network, newAdn=mutate_adn)
    network_1.Training(data=dataGen, p=200, dt=0.01, labels=None)
    #print("Accuracy mutate (1)=", network_1.generateEnergy(dataGen))
    #network_1.Training(data=dataGen, p=1900, dt=0.01, labels=None)
    #print("Accuracy mutate (2)=", network_1.generateEnergy(dataGen))
    network.Training(data=dataGen, p=100, dt=0.01, labels=None)

def Test_Mutacion():

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  50)
    #dataGen = GeneratorFromImage.GeneratorFromImage(2, 100)
    dataGen.dataConv2d()

    print("creating DNAs")
    space = DNA_test_f(dataGen.size[1], dataGen.size[2])

    for node in space.objects:

        parentDNA = space.node2key(node)
        if str(parentDNA) == str(space.center):
            print("PARENT DNA= ", parentDNA)
            network = nw_dendrites.Network(parentDNA, cudaFlag=True)
            network.Training(data=dataGen, p=300, dt=0.01, labels=None)

            for nodeKid in node.kids:

                kidDNA = space.node2key(nodeKid)
                print("KID DNA= ", kidDNA)
                network_kid = Mutate_Dendrites.executeMutation(oldNetwork=network, newAdn=kidDNA)
                network_kid.Training(data=dataGen, p=100, dt=0.01, labels=None)

#Test_pytorchNetwork()

#Test_Batch(dataGen)
#Test_Mutacion_dendrites()
Test_Mutacion()
