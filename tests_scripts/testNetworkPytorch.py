import children.pytorch.network_dendrites as nw_dendrites
from DAO import GeneratorFromImage, GeneratorFromCIFAR

from Geometric.Graphs.DNA_Graph import DNA_Graph
from Geometric.Creators.DNA_creators import Creator_from_selection_nm as Creator_nm
from utilities.Abstract_classes.classes.uniform_random_selector import(
    centered_random_selector as Selector_creator)
from Geometric.Conditions.DNA_conditions import max_layer,max_filter,max_filter_dense
import Geometric.Directions.DNA_directions_convex as direction_dna
import Geometric.Directions.DNA_directions_pool as direction_dna_p
import Geometric.Directions.DNA_directions_pool_duplicate as direction_dna_p_duplicate
import mutations.mutation_manager as MutationManager
import const.versions as directions_version

import os
import utilities.NetworkStorage as StorageManager
import utilities.ExperimentSettings as ExperimentSettings
import utilities.Augmentation as Augmentation
import utilities.AugmentationSettings as AugmentationSettings

import utilities.MemoryManager as MemoryManager
import tests_scripts.test_DNAs as test_DNAs
import torch
import time
import gc
import numpy as np
import math

def pcos(x):
    if x>np.pi:
        x-np.pi
    return np.cos(x)

def Alaising(M,m,ep):
    M=10**(-M)
    m=10**(-m)
    return [ m+1/2*(M-m)*(1+pcos(t/ep*np.pi))
             for t in range(0,ep)]

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

        nodeDna = space.node2key(node)

        if str(nodeDna) == str(space.center):
            nodeCenter = node

    return nodeCenter

def Test_Mutacion():
    memoryManager = MemoryManager.MemoryManager()

    augSettings = AugmentationSettings.AugmentationSettings()

    list_transform = { 
        augSettings.randomHorizontalFlip : True,
        augSettings.translate : True,
    }


    PARENT_DNA = ( (-1, 1, 3, 32, 32), (0, 3, 64, 3, 3),(0, 64, 128, 3, 3, 2),  (0, 192, 256, 3, 3, 2), (0, 256, 256, 13, 13), (1, 256, 10), (2,), 
(3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 0, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5) )


    MUTATE_DNA = direction_dna.spread_convex_dendrites(1, PARENT_DNA)
    print("MUTATED DNA: ", MUTATE_DNA)
    transform_compose = augSettings.generateTransformCompose(list_transform, False)
    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  64, threads=0, dataAugmentation=True, transforms_mode=transform_compose)
    dataGen.dataConv2d()
    
    version = directions_version.POOL_VERSION

    mutation_manager = MutationManager.MutationManager(directions_version=version)
    
    parent_network = nw_dendrites.Network(dna=PARENT_DNA, cuda_flag=True, momentum=0.9, weight_decay=0, 
                enable_activation=True, enable_track_stats=True, dropout_value=0.2, dropout_function=None, version=version)

    parent_network.training_cosine_dt(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=1, restart_dt=1)

    parent_network.generate_accuracy(dataGen)
    print("original acc: ", parent_network.get_accuracy())
    mutate_network = mutation_manager.execute_mutation(parent_network, MUTATE_DNA)
    mutate_network.generate_accuracy(dataGen)
    print("mutated acc: ", mutate_network.get_accuracy())
    mutate_network.training_cosine_dt(dataGenerator=dataGen, max_dt=0.001, min_dt=0.001, epochs=1, restart_dt=1)

    mutate_network.generate_accuracy(dataGen)
    print("mutated acc after training: ", mutate_network.get_accuracy())

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

    DNA = ((-1, 1, 3, 32, 32), (0, 3, 32, 3, 3), (0, 32, 64, 3, 3, 2), (0, 64, 128, 3, 3, 2), (0, 128, 256, 3, 3), (0, 256, 128, 2, 2), (0, 128, 128, 3, 3, 2), (0, 256, 256, 3, 3), (0, 384, 256, 3, 3), (0, 256, 128, 3, 3), (0, 384, 128, 3, 3), (0, 128, 128, 3, 3), (0, 256, 256, 3, 3), (0, 256, 128, 8, 8), (1, 128, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 3, 6), (3, 5, 7), (3, 6, 7), (3, 7, 8), (3, 8, 9), (3, 3, 9), (3, 4, 10), (3, 10, 11), (3, 9, 11), (3, 11, 12), (3, 13, 14))
    MUTATE_DNA = ((-1, 1, 3, 32, 32), (0, 3, 32, 3, 3), (0, 32, 64, 3, 3, 2), (0, 64, 128, 3, 3, 2), (0, 128, 256, 3, 3), (0, 256, 128, 2, 2), (0, 128, 128, 3, 3, 2), (0, 256, 256, 3, 3), (0, 128, 128, 3, 3), (0, 384, 256, 3, 3), (0, 256, 128, 3, 3), (0, 384, 128, 3, 3), (0, 128, 128, 3, 3), (0, 256, 256, 3, 3), (0, 256, 128, 8, 8), (1, 128, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 3, 6), (3, 5, 7), (3, 7, 8), (3, 6, 8), (3, 8, 9), (3, 9, 10), (3, 3, 10), (3, 4, 11), (3, 11, 12), (3, 10, 12), (3, 12, 13), (3, 14, 15))

    ((-1, 1, 3, 32, 32), (0, 3, 32, 3, 3), (0, 32, 64, 3, 3, 2), (0, 64, 128, 3, 3, 2), 
        (0, 128, 256, 3, 3), (0, 256, 256, 8, 8), (1, 128, 10), (2,), (3, -1, 0), 
        (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 5, 6))


    parent_network = nw_dendrites.Network(dna=DNA, cuda_flag=True, momentum=0.9, weight_decay=0, 
                enable_activation=True, enable_track_stats=True, dropout_value=0, dropout_function=None, version=version)   

    print("starting mutation")
    mutate_network = mutation_manager.execute_mutation(parent_network, MUTATE_DNA)


def TestMemoryManager():
    
    epochs = 0.2
    batch_size = 64

    def dropout_function(base_p, total_layers, index_layer, isPool=False):

        value = 0
        if index_layer != 0 and isPool == False:
            value = base_p

        if index_layer == total_layers - 2:
            value = base_p

        print("conv2d: ", index_layer, " - dropout: ", value, " - isPool: ", isPool)

        return value

    settings = ExperimentSettings.ExperimentSettings()
    settings.momentum = 0.9
    settings.dropout_value = 0.05
    settings.weight_decay = 0.0005
    settings.enable_activation = True
    settings.enable_last_activation = True
    settings.enable_track_stats = True
    settings.version = directions_version.CONVEX_VERSION
    settings.eps_batchorm = 0.001
    settings.dropout_function = dropout_function
    settings.ricap = Augmentation.Ricap(beta=0.3)

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  batch_size, threads=0, dataAugmentation=True)
    dataGen.dataConv2d()
    memoryManager = MemoryManager.MemoryManager()

    mutation_manager = MutationManager.MutationManager(directions_version=settings.version)

    dna = test_DNAs.DNA_base

    e =  50000 / batch_size
    e = math.ceil(e)
    print("e: ", e)
    print("total iterations: ", int(epochs*e))
    dt_array = Alaising(1.2, 99, int(epochs*e))

    input("press to continue: before load network")
    network = nw_dendrites.Network(dna, cuda_flag=True, momentum=settings.momentum, weight_decay=settings.weight_decay,
                                    enable_activation=settings.enable_activation,
                                    enable_track_stats=settings.enable_track_stats, dropout_value=settings.dropout_value,
                                    dropout_function=settings.dropout_function, enable_last_activation=settings.enable_last_activation,
                                    version=settings.version, eps_batchnorm=settings.eps_batchorm)

    
    input("press to continue: before training network")

    network.training_custom_dt(dataGenerator=dataGen,dt_array=dt_array, ricap=settings.ricap, evalLoss=True)
    
    network.generate_accuracy(dataGen)
    print("net acc: ", network.get_accuracy())

    input("press to continue: before save network")
    memoryManager.saveTempNetwork(network)
    input("press to continue: after save network")

    input("press to continue: before load temp network")
    network_loaded = memoryManager.loadTempNetwork(dna, settings)
    input("press to continue: after load temp network")
    
    network_loaded.generate_accuracy(dataGen)
    print("loaded acc: ", network_loaded.get_accuracy())

    input("press to continue: before mutate network (add filters layer 1)")
    dna_mutate = direction_dna.increase_filters(1, network_loaded.dna)
    network_mutate = mutation_manager.execute_mutation(network_loaded, dna_mutate)
    input("press to continue: after mutate network")
    
    input("press to continue: before delete old network")
    memoryManager.deleteNetwork(network_loaded)
    input("press to continue: after delete old network")

    network_mutate.generate_accuracy(dataGen)
    print("mutated acc: ", network_mutate.get_accuracy())
    input("press to conitnue: before training mutate network")
    network_mutate.training_custom_dt(dataGenerator=dataGen,dt_array=dt_array, ricap=settings.ricap, evalLoss=True)
    input("press to conitnue: after training mutate network")
    network_mutate.generate_accuracy(dataGen)
    print("mutate net acc: ", network_mutate.get_accuracy())

    input("press to continue: before save network")
    memoryManager.saveTempNetwork(network_mutate)
    input("press to continue: after save network")

    input("press to continue: before load network")
    network_loaded = memoryManager.loadTempNetwork(dna_mutate, settings)
    input("press to continue: after load network")

    network_loaded.generate_accuracy(dataGen)
    print("loaded acc: ", network_loaded.get_accuracy())

    input("press to continue: before mutate network (add layer pool 1)")
    dna_mutate_2 = direction_dna.add_pool_layer(1, network_loaded.dna)
    network_mutate = mutation_manager.execute_mutation(network_loaded, dna_mutate_2)
    input("press to continue: after mutate network")
    
    input("press to continue: before delete old network")
    memoryManager.deleteNetwork(network_loaded)
    input("press to continue: after delete old network")

    network_mutate.generate_accuracy(dataGen)
    print("mutated acc: ", network_mutate.get_accuracy())
    input("press to conitnue: before training mutate network")
    network_mutate.training_custom_dt(dataGenerator=dataGen,dt_array=dt_array, ricap=settings.ricap, evalLoss=True)
    input("press to conitnue: after training mutate network")
    network_mutate.generate_accuracy(dataGen)
    print("mutate net acc: ", network_mutate.get_accuracy())

    input("press to continue: before save network")
    memoryManager.saveTempNetwork(network_mutate)
    input("press to continue: after save network")

def Test_param_calculator():

    dna = test_DNAs.DNA_val_25 

    total_params = 0
    conv2d = 0
    for layer in dna:

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

    #init = 1593541402.0297086
    #final = 1593541564.913432

    init = 1593076326.1944137
    final = 1593076503.2711444
    time_200_mini_batches = (final - init) / 16
    print("time 200 minibatches (s): ", time_200_mini_batches)
    time_1_epoch = time_200_mini_batches * 782 / 47
    print("time 1 epoch (s): ", time_1_epoch)
    time_200_epochs = time_1_epoch * 200
    print("time 200 epochs (hr): ", time_200_epochs / 3600)
    time_400_epochs = time_1_epoch * 400
    print("time 400 epochs (hr): ", time_400_epochs / 3600)
    time_600_epochs = time_1_epoch * 600
    print("time 600 epochs (hr): ", time_600_epochs / 3600)


def LayerCalculator():

    DNA = ((-1, 1, 3, 32, 32), (0, 3, 128, 4, 4), (0, 128, 128, 3, 3, 2), (0, 128, 128, 2, 2), (0, 128, 256, 2, 2, 2), (0, 128, 128, 3, 3), (0, 128, 256, 3, 3), (0, 256, 512, 2, 2), (0, 256, 128, 3, 3), (0, 512, 512, 3, 3), (0, 512, 512, 3, 3), (0, 640, 512, 3, 3, 2), (0, 768, 512, 3, 3), (0, 640, 512, 4, 4, 2), (0, 512, 512, 4, 4), (0, 512, 512, 2, 2), (0, 512, 512, 3, 3), (0, 1024, 256, 3, 3), (0, 256, 256, 2, 2), (0, 256, 512, 3, 3), (0, 512, 256, 4, 4), (0, 512, 512, 5, 5), (0, 1024, 512, 2, 2), (0, 512, 512, 4, 4), (0, 512, 512, 3, 3), (0, 512, 512, 4, 4), (0, 512, 512, 2, 2), (0, 512, 512, 3, 3), (0, 512, 512, 3, 3), (0, 512, 512, 3, 3), (0, 1024, 512, 2, 2), (0, 512, 512, 3, 3), (0, 512, 256, 3, 3), (0, 512, 256, 4, 4), (0, 768, 256, 3, 3), (0, 256, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 2, 4), (3, 4, 5), (3, 3, 6), (3, 5, 7), (3, 3, 7), (3, 6, 8), (3, 8, 9), (3, 7, 10), (3, 9, 10), (3, 10, 11), (3, 3, 11), (3, 11, 12), (3, 1, 12), (3, 12, 13), (3, 13, 14), (3, 14, 15), (3, 10, 16), (3, 15, 16), (3, 3, 17), (3, 17, 18), (3, 18, 19), (3, 3, 19), (3, 16, 20), (3, 19, 20), (3, 20, 21), (3, 13, 21), (3, 21, 22), (3, 22, 23), (3, 23, 24), (3, 12, 24), (3, 24, 25), (3, 25, 26), (3, 3, 26), (3, 26, 27), (3, 27, 28), (3, 28, 29), (3, 12, 29), (3, 25, 30), (3, 29, 31), (3, 30, 31), (3, 31, 32), (3, 3, 32), (3, 32, 33), (3, 21, 33), (3, 33, 34), (3, 34, 35), (3, 35, 36))    
    
    
    count = 0
    for layer in DNA:

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