import children.pytorch.NetworkDendrites as nw_dendrites
from DAO import GeneratorFromImage, GeneratorFromCIFAR

from DNA_Graph import DNA_Graph
from DNA_creators import Creator_from_selection_nm as Creator_nm
from utilities.Abstract_classes.classes.uniform_random_selector import(
    centered_random_selector as Selector_creator)
from DNA_conditions import max_layer,max_filter,max_filter_dense
import DNA_directions_h as direction_dna
import children.pytorch.MutationManager as MutationManager
import const.versions as directions_version

import os
import utilities.NetworkStorage as StorageManager
import TestNetwork.ExperimentSettings as ExperimentSettings
import TestNetwork.AugmentationSettings as AugmentationSettings

import utilities.MemoryManager as MemoryManager
import test_DNAs
import torch

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

def generateSpace(x,y):
    max_layers=10
    max_filters=60
    max_dense=100
    def condition_b(z):
        return max_filter_dense(max_filter(max_layer(z,max_layers),max_filters),max_dense)
    center=((-1,1,3,32,32),
            (0,3, 5, 3 , 3),
            (0,5, 6, 3,  3),
            (0,6,7,3,3,2),
            (0,7, 8, 16,16),
            (1, 8,10),
             (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4),
            (3,4,5))
    version='h'
    mutations=((4,0,0,0),(1,0,0,0),(0,0,1))
    #mutations=((4,0,0,0),(1,0,0,0),(0,1,0,0),(0,1,0,0),(0,0,1))
    sel=Selector_creator(condition=condition_b,
        directions=version,mutations=mutations,num_actions=5)
    print('The selector is')
    print(sel)
    sel.update(center)
    actions=sel.get_predicted_actions()
    creator=Creator_nm
    space=DNA_Graph(center,1,(x,y),condition_b,actions,
        version,creator=creator,num_morphisms=5,selector=sel)
    return space

def getNodeCenter(space):

    nodeCenter = None
    for node in space.objects:

        nodeAdn = space.node2key(node)

        if str(nodeAdn) == str(space.center):
            nodeCenter = node

    return nodeCenter

def Test_Mutacion():

    space = generateSpace(32, 32)

    nodeCenter = getNodeCenter(space)

    #for nodeKid in nodeCenter.kids:
    #    print("path kid: ", nodeKid.objects[0].objects[0].path)

    PARENT_DNA = space.node2key(nodeCenter)

    augSettings = AugmentationSettings.AugmentationSettings()

    list_transform = { 
        augSettings.randomHorizontalFlip : True,
        augSettings.translate : True,
    }

    transform_compose = augSettings.generateTransformCompose(list_transform, False)
    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  128, threads=0, dataAugmentation=True, transforms_mode=transform_compose)
    dataGen.dataConv2d()
    
    version = directions_version.H_VERSION

    mutation_manager = MutationManager.MutationManager(directions_version=version)
    
    parent_network = nw_dendrites.Network(adn=PARENT_DNA, cudaFlag=True, momentum=0.9, weight_decay=0, 
                enable_activation=True, enable_track_stats=True, dropout_value=0.2, dropout_function=None, version=version)

    parent_network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=1, restart_dt=1, 
                                        show_accuarcy=True)
    parent_network.generateEnergy(dataGen)
    print("Parent ACC: ", parent_network.getAcurracy())    

    kid_num = 1
    for nodeKid in nodeCenter.kids:

        print("kid: ", kid_num)
        print("DNA KID: ", space.node2key(nodeKid))
        print("Las path: ", nodeKid.objects[0].objects[0].path[-1])
        mutations_path =  nodeKid.objects[0].objects[0].path

        kid_dna_num = 1
        old_network = parent_network

        for kid_dna in mutations_path:

            mutate_network = mutation_manager.executeMutation(old_network, kid_dna)
            mutate_network.generateEnergy(dataGen)
            print("kid ACC: ", mutate_network.getAcurracy())
            old_network = mutate_network

            kid_dna_num += 1

        print("")

        kid_num += 1
        

    #mutate_network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=1, restart_dt=1, show_accuarcy=True)
    
def TestMemoryManager():
    
    settings = ExperimentSettings.ExperimentSettings()
    settings.momentum = 0.9
    settings.dropout_value = 0
    settings.weight_decay = 0.0005
    settings.enable_activation = True
    settings.enable_last_activation = False
    settings.enable_track_stats = True
    settings.version = directions_version.H_VERSION
    
    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  128, threads=0, dataAugmentation=True)
    dataGen.dataConv2d()
    memoryManager = MemoryManager.MemoryManager()

    adn = test_DNAs.DNA_calibration_3

    input("press to continue: before load network")
    network = nw_dendrites.Network(adn, cudaFlag=True, momentum=settings.momentum, weight_decay=settings.weight_decay,
                                    enable_activation=settings.enable_activation, enable_track_stats=settings.enable_track_stats,
                                    dropout_value=settings.dropout_value, enable_last_activation=settings.enable_last_activation,
                                    version=settings.version)

    
    input("press to continue: before training network")
    network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=1, restart_dt=1, 
                                        show_accuarcy=True)

    network.generateEnergy(dataGen)
    print("net acc: ", network.getAcurracy())

    input("press to continue: before save network")
    memoryManager.saveTempNetwork(network)
    input("press to continue: after save network")
    
    if network == None:
        print("network = None")

    input("press to continue: before load temp network")
    network_loaded = memoryManager.loadTempNetwork(adn, settings)

    input("press to continue: after load temp network")
    if network_loaded != None:
        print("network loaded")
        network_loaded.generateEnergy(dataGen)
        print("loaded acc: ", network_loaded.getAcurracy())
        input("press to conitnue: before training network")
        network_loaded.TrainingCosineLR_Restarts(dataGenerator=dataGen, 
                                        max_dt=0.001, min_dt=0.001, epochs=1, restart_dt=1,        
                                        show_accuarcy=True)
        
        input("press to conitnue: after training network")
        network_loaded.generateEnergy(dataGen)
        print("net acc: ", network_loaded.getAcurracy())
        input("press to continue: before save network")
        memoryManager.saveTempNetwork(network_loaded)
        input("press to continue: after save network")

    

if __name__ == "__main__":
    #Test_Mutacion()
    TestMemoryManager()