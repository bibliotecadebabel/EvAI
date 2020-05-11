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

from DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector


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

def DNA_Creator_s(x,y):
    def condition(DNA):
        output=True
        if DNA:
            for num_layer in range(0,len(DNA)-3):
                layer=DNA[num_layer]
                
                if layer[0] == 0:
                    x_l=layer[3]
                    y_l=layer[4]
                    output=output and (x_l<x) and (y_l<y)
        if output:
            return max_layer(DNA,15)
    #center=((-1, 1, 3, 11, 11), (0, 3, 5, 3, 3), (0, 5, 5, 3, 3), (0, 5, 5, 7, 7), (1, 5, 10), (2,), (3, -1, 0),(3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4))
    center=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0,5, 5, x-2, y-2),
            (1, 5, 10), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3))
    selector=random_selector()
    selector.update(center)
    actions=((0, (0,1,0,0)), (1, (0,1,0,0)), (0, (0,-1,0,0)), (1, (0,-1,0,0)), (0, (-1,0,0,0)))
    #actions=((1, (-1,0,0,0)), )
    version='final'
    space=DNA_Graph(center,1,(x,y),condition,actions, version,Creator_s)

    
    return space

def Test_Mutacion_dendrites():

    indexTarget = 0

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  50)
    #dataGen = GeneratorFromImage.GeneratorFromImage(2, 100)
    dataGen.dataConv2d()

    print("creating DNAs")

    #dna_list = spread_dendrites_test(dataGen.size[1], dataGen.size[2], indexTarget)

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

    #dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  50)
    dataGen = GeneratorFromImage.GeneratorFromImage(2, 100)
    dataGen.dataConv2d()

    print("creating DNAs")
    space = DNA_Creator_s(dataGen.size[1], dataGen.size[2])

    for node in space.objects:

        parentDNA = space.node2key(node)
        if str(parentDNA) == str(space.center):
            print("PARENT DNA= ", parentDNA)
            network = nw_dendrites.Network(parentDNA, cudaFlag=True)
            network.Training(data=dataGen, p=1000, dt=0.01, labels=None)

            for nodeKid in node.kids:

                kidDNA = space.node2key(nodeKid)
                print("KID DNA= ", kidDNA)
                network_kid = Mutate_Dendrites.executeMutation(oldNetwork=network, newAdn=kidDNA)
                network_kid.Training(data=dataGen, p=100, dt=0.01, labels=None)

#Test_pytorchNetwork()

#Test_Batch(dataGen)
#Test_Mutacion_dendrites()
Test_Mutacion()
