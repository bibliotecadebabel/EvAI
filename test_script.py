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
    value = 0
    if index_conv2d == 2:
        value = 0.2
    if index_conv2d == 4:
        value = 0.3
    if index_conv2d == 6:
        value = 0.4
    if index_conv2d == 7:
        value = 0.5

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

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  64, threads=2, dataAugmentation=True)
    dataGen.dataConv2d()
    
    adn = ((-1, 1, 3, 32, 32), (0, 3, 32, 3, 3),(0, 32, 32, 3, 3), (0, 32, 64, 3, 3, 2), 
            (0, 64, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 128, 128, 3, 3), (0, 128, 128, 1, 1, 2), (1, 128, 10), (2,), 
            (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 6, 7), (3, 7, 8))

    network = nw_dendrites.Network(adn=adn, cudaFlag=True, momentum=0.9, weight_decay=0, 
                enable_activation=True, enable_track_stats=True, dropout_value=0.2, dropout_function=dropout_function)

    network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=400, restart_dt=400, show_accuarcy=True)



if __name__ == "__main__":
    Test_Mutacion()
