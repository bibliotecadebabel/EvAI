import children.pytorch.NetworkDendrites as nw_dendrites
from DAO import GeneratorFromImage, GeneratorFromCIFAR

from DNA_Graph import DNA_Graph
from DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector
from utilities.Abstract_classes.classes.positive_random_selector import(
    centered_random_selector as Selector_creator)
from DNA_conditions import max_layer,max_filter

from DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector
import DNA_directions_h as direction_dna
import children.pytorch.MutationManager as MutationManager
import const.versions as directions_version

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

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  128, threads=2, dataAugmentation=True)
    dataGen.dataConv2d()
    
    version = directions_version.H_VERSION

    mutation_manager = MutationManager.MutationManager(directions_version=version)
    
    DNA =  ((-1,1,3,32,32),(0, 3, 5, 3, 3),(0, 5, 8,2,2,2),(0,8,120,16,16),
            (1, 120, 10),
            (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4))

    mutate_DNA = direction_dna.increase_kernel(1, DNA)

    print("DNA: ", DNA)
    print("new DNA: ", mutate_DNA)
    network = nw_dendrites.Network(adn=DNA, cudaFlag=True, momentum=0.9, weight_decay=0, 
            enable_activation=True, enable_track_stats=True, dropout_value=0.2, dropout_function=None, version=version)
    
    network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=5, restart_dt=5, show_accuarcy=True)

    network.generateEnergy(dataGen)
    print("Original accuracy: ", network.getAcurracy())

    mutate_network = mutation_manager.executeMutation(network, mutate_DNA)
    mutate_network.generateEnergy(dataGen)
    print("Accuracy after mutation: ", mutate_network.getAcurracy())

    mutate_network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=5, restart_dt=5, show_accuarcy=True)
    mutate_network.generateEnergy(dataGen)
    print("Accuracy after mutation training: ", mutate_network.getAcurracy())
    #mutate_network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=1, restart_dt=1, show_accuarcy=True)

    

if __name__ == "__main__":
    Test_Mutacion()
