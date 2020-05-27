import children.pytorch.Network as nw
import children.pytorch.NetworkDendrites as nw_dendrites
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
import children.pytorch.MutateNetwork as MutateNetwork
import children.pytorch.MutateNetwork_Dendrites_clone as Mutate_Dendrites
from DAO import GeneratorFromImage, GeneratorFromCIFAR

from DNA_Graph import DNA_Graph
from DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector
from utilities.Abstract_classes.classes.positive_random_selector import(
    centered_random_selector as Selector_creator)
from DNA_conditions import max_layer,max_filter

from DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector
import DNA_directions_pool as direction_dna

def dropout_function(base_p, total_conv2d, index_conv2d):
    
    print("index: ", index_conv2d)
    value = base_p / (total_conv2d - index_conv2d)
    value = value * 2

    return value

def DNA_pool(x,y):
    max_layers=10
    max_filters=60
    def condition_b(z):
        return max_filter(max_layer(z,max_layers),max_filters)
    center=((-1,1,3,x,y),
            (0,3, 15, 3 , 3),
            (0,15, 15, 3,  3),
            (0,15,33,2,2,2),
            (0,33, 50, 13, 13),
            (1, 50,10),
             (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4),
            (3,4,5))
    version='pool'
    mutations=((4,0,0,0),(1,0,0,0),(0,1,0,0),(0,1,0,0))
    selector=Selector_creator(condition=condition_b,
        directions=version,mutations=mutations,num_actions=10)
    selector.update(center)
    actions=selector.get_predicted_actions()
    creator=Creator_s
    space=DNA_Graph(center,1,(x,y),condition_b,actions,
        version,creator)
    return space


def Test_Mutacion():

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  128, threads=2, dataAugmentation=True)
    dataGen.dataConv2d()

    print("creating DNAs")

    parent = ((-1, 1, 3, 32, 32), (0, 3, 16, 4, 4), (0, 16, 32, 3, 3), (0, 48, 32, 3, 3, 2), (0, 32, 32, 3, 3, 2), 
                    (0, 64, 64, 3, 3, 2), (0, 96, 64, 2, 2, 2), (0, 128, 128, 6, 6), (1, 128, 10), (2,), 
                        (3, -1, 0), (3, 0, 1), (3, 0, 2), (3, 1, 2), (3, 2, 3), (3, 2, 4), (3, 3, 4),
                         (3, 4, 5), (3, 2, 5), (3, 4, 6), (3, 5, 6), (3, 6, 7), (3, 7, 8))

    child = direction_dna.spread_dendrites(3, parent)
    network = nw_dendrites.Network(child, cudaFlag=True, momentum=0.9, weight_decay=0.0, 
                                    enable_activation=True, enable_track_stats=True, dropout_value=0.20)
        
    network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, 
                    epochs=1, restart_dt=1, show_accuarcy=True)
    
    print("child: ", child)
    
    '''
    space = DNA_pool(32, 32)
    
    for node in space.objects:
        
        parentDNA = space.node2key(node)

        print("current DNA: ", parentDNA)
        network = nw_dendrites.Network(parentDNA, cudaFlag=True, momentum=0.9, weight_decay=0.0, 
                                    enable_activation=True, enable_track_stats=True, dropout_value=0.20)
        
        network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, 
                    epochs=1, restart_dt=1, show_accuarcy=True)

        network.generateEnergy(dataGen)
        print("accuracy: ", network.getAcurracy())
        
        for nodek in node.kids:
            
            mutate_dna = space.node2key(nodek)
            print("mutate dna: ", mutate_dna)
            network_kid =  Mutate_Dendrites.executeMutation(network, mutate_dna)
            network_kid.generateEnergy(dataGen)
            print("accuracy after mutate: ", network_kid.getAcurracy())
            network_kid.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, 
                                                epochs=1, restart_dt=1, show_accuarcy=True)
            network_kid.generateEnergy(dataGen)
            print("accuracy after training mutate: ", network_kid.getAcurracy())

    #mutate_network = Mutate_Dendrites.executeMutation(network, mutateDNA)
    #mutate_network.generateEnergy(dataGen)
    #print("accuracy after mutate= ", mutate_network.getAcurracy())
    
    '''
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
