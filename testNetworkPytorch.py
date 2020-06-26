import children.pytorch.NetworkDendrites as nw_dendrites
from DAO import GeneratorFromImage, GeneratorFromCIFAR

from DNA_Graph import DNA_Graph
from DNA_creators import Creator_from_selection_nm as Creator_nm
from utilities.Abstract_classes.classes.uniform_random_selector import(
    centered_random_selector as Selector_creator)
from DNA_conditions import max_layer,max_filter,max_filter_dense
import DNA_directions_convex as direction_dna
import children.pytorch.MutationManager as MutationManager
import const.versions as directions_version

import os
import utilities.NetworkStorage as StorageManager
import TestNetwork.ExperimentSettings as ExperimentSettings
import TestNetwork.AugmentationSettings as AugmentationSettings

import utilities.MemoryManager as MemoryManager
import test_DNAs
import torch
import time
import gc

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
    memoryManager = MemoryManager.MemoryManager()

    augSettings = AugmentationSettings.AugmentationSettings()

    list_transform = { 
        augSettings.randomHorizontalFlip : True,
        augSettings.translate : True,
    }

    PARENT_DNA = ((-1, 1, 3, 32, 32), (0, 3, 64, 4, 4), (0, 64, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 128, 512, 3, 3, 2), (0, 512, 512, 4, 4), (0, 512, 128, 3, 3), (0, 256, 128, 3, 3), (0, 128, 128, 3, 3), (0, 256, 512, 3, 3, 2), (0, 1024, 512, 3, 3), (0, 512, 512, 3, 3), (0, 512, 512, 3, 3), (0, 512, 512, 3, 3), (0, 128, 128, 3, 3), (0, 640, 512, 2, 2), (0, 512, 256, 3, 3), (0, 768, 512, 4, 4, 2), (0, 512, 512, 3, 3), (0, 1024, 512, 4, 4), (0, 640, 512, 3, 3), (0, 512, 512, 3, 3), (0, 512, 512, 4, 4), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 2, 6), (3, 6, 7), (3, 2, 8), (3, 7, 8), (3, 8, 9), (3, 4, 9), (3, 9, 10), (3, 3, 11), (3, 10, 12), (3, 11, 12), (3, 2, 13), (3, 12, 14), (3, 13, 14), (3, 2, 15), (3, 3, 15), (3, 14, 16), (3, 15, 16), (3, 12, 17), (3, 17, 18), (3, 16, 18), (3, 18, 19), (3, 5, 19), (3, 16, 20), (3, 19, 21), (3, 20, 21), (3, 21, 22), (3, 22, 23), (3, 23, 24))
    MUTATE_DNA = ((-1, 1, 3, 32, 32), (0, 3, 64, 4, 4), (0, 64, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 128, 512, 3, 3, 2), (0, 512, 512, 4, 4), (0, 512, 128, 3, 3), (0, 256, 128, 3, 3), (0, 128, 128, 3, 3), (0, 256, 512, 3, 3, 2), (0, 1024, 512, 3, 3), (0, 512, 512, 3, 3), (0, 512, 512, 3, 3), (0, 512, 512, 3, 3), (0, 128, 128, 3, 3), (0, 640, 512, 2, 2), (0, 128, 256, 3, 3), (0, 768, 512, 4, 4, 2), (0, 512, 512, 3, 3), (0, 1024, 512, 4, 4), (0, 640, 512, 3, 3), (0, 512, 512, 3, 3), (0, 512, 512, 4, 4), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 2, 6), (3, 6, 7), (3, 2, 8), (3, 7, 8), (3, 8, 9), (3, 4, 9), (3, 9, 10), (3, 3, 11), (3, 10, 12), (3, 11, 12), (3, 2, 13), (3, 12, 14), (3, 13, 14), (3, 2, 15), (3, 14, 16), (3, 15, 16), (3, 12, 17), (3, 17, 18), (3, 16, 18), (3, 18, 19), (3, 5, 19), (3, 16, 20), (3, 19, 21), (3, 20, 21), (3, 21, 22), (3, 22, 23), (3, 23, 24))

    #transform_compose = augSettings.generateTransformCompose(list_transform, False)
    #dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  64, threads=0, dataAugmentation=True, transforms_mode=transform_compose)
    #dataGen.dataConv2d()
    
    version = directions_version.CONVEX_VERSION

    mutation_manager = MutationManager.MutationManager(directions_version=version)
    
    parent_network = nw_dendrites.Network(adn=PARENT_DNA, cudaFlag=True, momentum=0.9, weight_decay=0, 
                enable_activation=True, enable_track_stats=True, dropout_value=0.2, dropout_function=None, version=version)

    mutate_network = mutation_manager.executeMutation(parent_network, MUTATE_DNA)


    #mutate_network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=1, restart_dt=1, show_accuarcy=True)

def Test_Convex():
    augSettings = AugmentationSettings.AugmentationSettings()

    list_transform = { 
        augSettings.randomHorizontalFlip : True,
        augSettings.translate : True,
    }

    version = directions_version.CONVEX_VERSION

    mutation_manager = MutationManager.MutationManager(directions_version=version)
    transform_compose = augSettings.generateTransformCompose(list_transform, False)
    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  128, threads=0, dataAugmentation=True, transforms_mode=transform_compose)
    dataGen.dataConv2d()

    ADN = ((-1, 1, 3, 32, 32), (0, 3, 32, 3, 3), (0, 32, 64, 3, 3, 2), (0, 64, 128, 3, 3, 2), (0, 128, 256, 3, 3), (0, 256, 128, 2, 2), (0, 128, 128, 3, 3, 2), (0, 256, 256, 3, 3), (0, 384, 256, 3, 3), (0, 256, 128, 3, 3), (0, 384, 128, 3, 3), (0, 128, 128, 3, 3), (0, 256, 256, 3, 3), (0, 256, 128, 8, 8), (1, 128, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 3, 6), (3, 5, 7), (3, 6, 7), (3, 7, 8), (3, 8, 9), (3, 3, 9), (3, 4, 10), (3, 10, 11), (3, 9, 11), (3, 11, 12), (3, 13, 14))
    MUTATE_DNA = ((-1, 1, 3, 32, 32), (0, 3, 32, 3, 3), (0, 32, 64, 3, 3, 2), (0, 64, 128, 3, 3, 2), (0, 128, 256, 3, 3), (0, 256, 128, 2, 2), (0, 128, 128, 3, 3, 2), (0, 256, 256, 3, 3), (0, 128, 128, 3, 3), (0, 384, 256, 3, 3), (0, 256, 128, 3, 3), (0, 384, 128, 3, 3), (0, 128, 128, 3, 3), (0, 256, 256, 3, 3), (0, 256, 128, 8, 8), (1, 128, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 3, 6), (3, 5, 7), (3, 7, 8), (3, 6, 8), (3, 8, 9), (3, 9, 10), (3, 3, 10), (3, 4, 11), (3, 11, 12), (3, 10, 12), (3, 12, 13), (3, 14, 15))

    ((-1, 1, 3, 32, 32), (0, 3, 32, 3, 3), (0, 32, 64, 3, 3, 2), (0, 64, 128, 3, 3, 2), 
        (0, 128, 256, 3, 3), (0, 256, 256, 8, 8), (1, 128, 10), (2,), (3, -1, 0), 
        (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 5, 6))


    parent_network = nw_dendrites.Network(adn=ADN, cudaFlag=True, momentum=0.9, weight_decay=0, 
                enable_activation=True, enable_track_stats=True, dropout_value=0, dropout_function=None, version=version)   

    print("starting mutation")
    mutate_network = mutation_manager.executeMutation(parent_network, MUTATE_DNA)


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

    mutation_manager = MutationManager.MutationManager(directions_version=settings.version)

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

    input("press to continue: before load temp network")
    network_loaded = memoryManager.loadTempNetwork(adn, settings)
    input("press to continue: after load temp network")
    
    network_loaded.generateEnergy(dataGen)
    print("loaded acc: ", network_loaded.getAcurracy())

    input("press to continue: before mutate network (remove layer 1)")
    dna_mutate = direction_dna.remove_layer(1, network_loaded.adn)
    network_mutate = mutation_manager.executeMutation(network_loaded, dna_mutate)
    input("press to continue: after mutate network")
    
    input("press to continue: before delete old network")
    memoryManager.deleteNetwork(network_loaded)
    input("press to continue: after delete old network")

    network_mutate.generateEnergy(dataGen)
    print("mutated acc: ", network_mutate.getAcurracy())
    input("press to conitnue: before training mutate network")
    network_mutate.TrainingCosineLR_Restarts(dataGenerator=dataGen, 
                                    max_dt=0.001, min_dt=0.001, epochs=1, restart_dt=1,        
                                    show_accuarcy=True)
    input("press to conitnue: after training mutate network")
    network_mutate.generateEnergy(dataGen)
    print("mutate net acc: ", network_mutate.getAcurracy())

    input("press to continue: before save network")
    memoryManager.saveTempNetwork(network_mutate)
    input("press to continue: after save network")

    input("press to continue: before load network")
    network_loaded = memoryManager.loadTempNetwork(dna_mutate, settings)
    input("press to continue: after load network")

    network_loaded.generateEnergy(dataGen)
    print("loaded acc: ", network_loaded.getAcurracy())

    input("press to continue: before mutate network (remove layer 1)")
    dna_mutate_2 = direction_dna.remove_layer(1, network_loaded.adn)
    network_mutate = mutation_manager.executeMutation(network_loaded, dna_mutate_2)
    input("press to continue: after mutate network")
    
    input("press to continue: before delete old network")
    memoryManager.deleteNetwork(network_loaded)
    input("press to continue: after delete old network")

    network_mutate.generateEnergy(dataGen)
    print("mutated acc: ", network_mutate.getAcurracy())
    input("press to conitnue: before training mutate network")
    network_mutate.TrainingCosineLR_Restarts(dataGenerator=dataGen, 
                                    max_dt=0.001, min_dt=0.001, epochs=1, restart_dt=1,        
                                    show_accuarcy=True)
    input("press to conitnue: after training mutate network")
    network_mutate.generateEnergy(dataGen)
    print("mutate net acc: ", network_mutate.getAcurracy())

    input("press to continue: before save network")
    memoryManager.saveTempNetwork(network_mutate)
    input("press to continue: after save network")

def Test_param_calculator():

    adn = test_DNAs.DNA_val_20_ac_b 

    total_params = 0
    conv2d = 0
    for layer in adn:

        layer_type = layer[0]

        if layer_type == 0:
            conv2d += 1

            if len(layer) < 6:
                print("Layer type: ", layer_type, " - ", layer[1:5])
                param_per_layer = 1
                for param in layer[1:5]:
                    param_per_layer *= param
                
                #bias
                param_per_layer += layer[2]

                print("Layer params: ", param_per_layer)
                total_params += param_per_layer
                
        elif layer_type == 1:
            print("Layer type: ", layer_type)
            param_per_layer = 1
            for param in layer[1:]:
                param_per_layer *= param
            print("Layer params: ", param_per_layer)
            total_params += param_per_layer
        
    print("total params: ", total_params)  
    print("total conv2d: ", conv2d)

def TimeCalculator():

    start_time = 1592834371.4477036
    pretraining_end_time = 1592834714.5620644
    mutation_end_time = 1592843276.0939608

    #start_time_2 = 1593075099.9702818
    #mutation_end_time_2 = 1593076326.8922641

    pretraining_time = (pretraining_end_time - start_time)  / 60
    mutation_time = (mutation_end_time - pretraining_end_time) / 3600
    total_time = (mutation_end_time - start_time) / 3600

    #mutation_time = ((mutation_end_time - pretraining_end_time) + (mutation_end_time_2 - start_time_2)) / 3600
    #total_time = ((mutation_end_time - start_time) + (mutation_end_time_2 - start_time_2)) / 3600

    #total_time = (mutation_end_time - start_time) / 3600
    date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print("Test Date: ", date_time)
    print("pre training time: ", pretraining_time)
    print("mutation time: ", mutation_time)
    print("total time: ", total_time)

def TimeCalculator2():

    init = 1592860678.2520623
    final = 1592861146.3151975

    time_200_mini_batches = (final - init) / 16
    print("time 200 minibatches: ", time_200_mini_batches)
    time_1_epoch = time_200_mini_batches * 782 / 200
    print("time 1 epoch: ", time_1_epoch)
    time_600_epochs = time_1_epoch * 600
    print("time 600 epochs: ", time_600_epochs)
def LayerCalculator():

    ADN = ((-1, 1, 3, 32, 32), (0, 3, 512, 4, 4), (0, 512, 128, 3, 3), (0, 128, 512, 3, 3, 2), (0, 512, 512, 2, 2), (0, 512, 256, 3, 3, 2), (0, 256, 256, 3, 3), (0, 256, 512, 3, 3), (0, 512, 512, 3, 3), (0, 256, 256, 3, 3), (0, 768, 512, 3, 3), (0, 256, 256, 3, 3), (0, 256, 256, 3, 3), (0, 768, 512, 4, 4), (0, 768, 512, 2, 2), (0, 512, 512, 3, 3), (0, 512, 512, 3, 3), (0, 512, 512, 4, 4), (0, 512, 512, 3, 3), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, -1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 6, 7), (3, 4, 8), (3, 7, 9), (3, 8, 9), (3, 4, 10), (3, 10, 11), (3, 9, 12), (3, 11, 12), (3, 12, 13), (3, 4, 13), (3, 13, 14), (3, 14, 15), (3, 4, 15), (3, 15, 16), (3, 16, 17), (3, 17, 18), (3, 18, 19), (3, 19, 20))
    count = 0
    for layer in ADN:

        if layer[0] == 0:
            count += 1

    print("Layers: ", count)


if __name__ == "__main__":
    #Test_Mutacion()
    #TestMemoryManager()
    #Test_Convex()
    Test_param_calculator()
    #TimeCalculator()
    #LayerCalculator()
    #print(test_DNAs.DNA_val_20)