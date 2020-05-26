import children.pytorch.Network as nw
import children.pytorch.NetworkDendrites as nw_dendrites
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
import children.pytorch.MutateNetwork as MutateNetwork
import children.pytorch.MutateNetwork_Dendrites_clone as Mutate_Dendrites
from DAO import GeneratorFromImage, GeneratorFromCIFAR

from DNA_conditions import max_layer
from DNA_creators import Creator
from DNA_Graph import DNA_Graph

from DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector
import DNA_directions_clone as direction_dna


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


def Test_Mutacion():

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  64, threads=2)
    dataGen.dataConv2d()

    print("creating DNAs")

    parentDNA = ((-1,1,3,32,32),(0,3, 32, 3 , 3),(0,32, 32, 3,  3), (0,32, 64, 3, 3), 
                (0,64, 64, 3, 3), (0,64, 128, 24, 24), (1, 128,10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2),
                (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6))

    network = nw_dendrites.Network(parentDNA, cudaFlag=True, momentum=0.9, weight_decay=0.001, 
                                    enable_activation=True, enable_track_stats=True, dropout_value=0.10)

    network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=50, restart_dt=50, show_accuarcy=True)
    network.generateEnergy(dataGen)
    print("accuracy= ", network.getAcurracy())


def Test_Save_Model():
    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  50)
    dataGen.dataConv2d()
    parentDNA = ((-1,1,3,32,32),(0, 3, 5, 3, 3),(0,5, 5, 32-2, 32-2),
            (1, 5, 10), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3))

    network = nw_dendrites.Network(parentDNA, cudaFlag=True)
    network.Training(data=dataGen, p=2, dt=0.01, labels=None, full_database=True)
    network.generateEnergy(dataGen)
    print("Accuracy before save=", network.getAcurracy())

    network.saveModel("models/test_1")

def Test_Load_Model():

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  25)
    dataGen.dataConv2d()

    parentDNA =  ((-1, 1, 3, 32, 32), (0, 3, 60, 3, 3), (0, 63, 15, 3, 3), (0, 78, 15, 32, 32), (1, 15, 10), (2,), (3, -1, 0), (3, 0, 1), (3, -1, 1), (3, 1, 2), (3, 0, 2), (3, -1, 2), (3, 2, 3), (3, 3, 4))

    network = nw_dendrites.Network(parentDNA, cudaFlag=True)
    network.loadModel("saved_models/cifar/21_cifar-duplicate-restarts_model_4")
    network.generateEnergy(dataGen)
    print("Accuracy now=", network.getAcurracy())
    network.TrainingCosineLR(dataGenerator=dataGen, epochs=1, dt=0.1)
    network.generateEnergy(dataGen)
    print("Accuracy after training again=", network.getAcurracy())



if __name__ == "__main__":
    
    #Test_pytorchNetwork()

    #Test_Batch(dataGen)
    #Test_Mutacion_dendrites()
    Test_Mutacion()
    #Test_Save_Model()
    #Test_Load_Model()
