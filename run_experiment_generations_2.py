from TestNetwork.commands import CommandCreateDataGen
from TestNetwork.commands import  CommandExperimentCifar_Shuffle_generations as CommandExperimentCifar_Restarts
from DNA_conditions import max_layer,max_filter,max_filter_dense
import DNA_conditions
from DNA_creators import Creator
from DNA_Graph import DNA_Graph
from DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.uniform_random_selector_2 import centered_random_selector as random_selector
import TestNetwork.ExperimentSettings as ExperimentSettings
import const.versions as directions_version
import numpy as np
import test_DNAs as DNAs
import utilities.Augmentation as Augmentation
import TestNetwork.AugmentationSettings as AugmentationSettings
from utilities.Abstract_classes.classes.Alaising_cosine import (
    Damped_Alaising as Alai_w)
import math
import children.pytorch.NetworkDendrites as nw
import children.pytorch.MutationManager as MutationManager
###### EXPERIMENT SETTINGS ######

"""
def dropout_function(base_p, total_conv2d, index_conv2d, isPool=False):
    if index_conv2d==0:
        return 0
    else:
        value = base_p / (total_conv2d - index_conv2d)+base_p/2
    #print("conv2d: ", index_conv2d, " - dropout: ", value)
    return value
"""
'''
def dropout_function(base_p, total_conv2d, index_conv2d, isPool=False):
    if index_conv2d==0:
        return 0
    else:
        value = 0.1
        value = base_p / (total_conv2d - index_conv2d)+base_p/2
    #print("conv2d: ", index_conv2d, " - dropout: ", value)
    return value
'''
def dropout_function(base_p, total_layers, index_layer, isPool=False):

    value = 0
    if index_layer != 0 and isPool == False:
        value = base_p

    if index_layer == total_layers - 2:
        value = base_p

    print("conv2d: ", index_layer, " - dropout: ", value)

    return value

"""
def dropout_function(base_p, total_conv2d, index_conv2d):
    value = base_p +(3/5*base_p-base_p)*(total_conv2d - index_conv2d)/total_conv2d
    #print("conv2d: ", index_conv2d, " - dropout: ", value)
    return value
"""
'''
def dropout_function(base_p, total_layers, index_layer, isPool=False):

    value = 0
    if index_layer != 0 and isPool == False:
        value = base_p +(3/5*base_p-base_p)*(total_layers - index_layer-1)/total_layers

    if index_layer == total_layers - 2:
        value = base_p +(3/5*base_p-base_p)*(total_layers - index_layer-1)/total_layers

    #print("conv2d: ", index_layer, " - dropout: ", value)

    return value
'''
"""
def dropout_function(base_p, total_layers, index_layer, isPool=False):

    value = 0
    if index_layer != 0 and isPool == False:
        value = base_p

    if index_layer == total_layers - 2:
        value = base_p

    print("conv2d: ", index_layer, " - dropout: ", value)

    return value
"""


def pcos(x):
    if x>np.pi:
        x-np.pi
    return np.cos(x)

def Alaising(M,m,ep):
    M=10**(-M)
    m=10**(-m)
    return [ m+1/2*(M-m)*(1+pcos(t/ep*np.pi))
             for t in range(0,ep)]

def exp_alai(r,I,t_M,t_m):
    return [ 10 ** (-t_M-((t_m-t_M)*(k  %  int(I*r) )/I*r)) for k in range(I)]

def DNA_Creator_s(x,y, dna, version):
    def condition(DNA):
        return DNA_conditions.dict2condition(DNA,list_conditions)

    selector = None
    selector=random_selector(condition=condition,
        directions=version, num_actions=num_actions,
        mutations=(
        (0,1,0,0),(1,0,0,0),
        (0,0,1),(4,0,0,0),
        ))
    selector.update(dna)
    actions=selector.get_predicted_actions()
    #actions = ((0, (0,1,0,0)), (1, (0,1,0,0)), (0, (1,0,0,0)))
    space=DNA_Graph(dna,1,(x,y),condition,actions
        ,version,Creator_s)

    return [space, selector]

if __name__ == '__main__':

    settings = ExperimentSettings.ExperimentSettings()

    augSettings = AugmentationSettings.AugmentationSettings()

    dict_transformations = {
        augSettings.baseline_customRandomCrop : True,
        augSettings.randomHorizontalFlip : True,
        augSettings.randomErase_1 : True,
    }

    transform_compose = augSettings.generateTransformCompose(dict_transformations, False)
    # DIRECTIONS VERSION
    settings.version = 'convex'
    # NUM OF THREADS
    THREADS = int(input("Enter threads: "))

    # BATCH SIZE
    settings.batch_size = int(input("Enter batchsize: "))

    e =  50000 / settings.batch_size
    e = math.ceil(e)
    print("e = ", e)

    # DATA SOURCE ('default' -> Pikachu, 'cifar' -> CIFAR)
    DATA_SOURCE = 'cifar'

    # CUDA parameter (true/false)
    settings.cuda = True

    # Every PERDIOD_SAVE iterations, all DNA and its energies will be stored in the database.
    settings.period_save_space = 1

    # Every PERDIOD_NEWSPACE iterations, a new DNA GRAPH will be generated with the dna of the lowest energy network as center.
    settings.period_new_space = 1

    # Every PERIOD_SAVE_MODEL iterations, the best network (current center) will be stored on filesystem
    settings.period_save_model = 1

    # EPOCHS
    settings.epochs = int(input("Enter amount of epochs: "))

    settings.eps_batchorm = 0.001

    # INITIAL DT PARAMETERS
    num_actions=8
    settings.save_txt = True
    settings.max_init_iter = 360
    INIT_ITER = 200*e
    #alai_w=Alai_w(initial_max=0.1,final_max=0.05,
    #    initial_min=10**(-99),final_min=10**(-99),Max_iter=INIT_ITER)
    #settings.init_dt_array = alai_w.get_increments(225*e)
    #settings.init_dt_array = exp_alai(1,INIT_ITER,1,7)
    settings.init_dt_array =  Alaising(1.2,99,INIT_ITER)


    # JOINED DT PARAMETERS
    JOINED_ITER = 0*e
    #settings.joined_dt_array = Alaising(2,6,e)
    settings.joined_dt_array = Alaising(1,5,JOINED_ITER)
    settings.max_joined_iter = 1

    # BEST DT PARAMETERS
    BEST_ITER = 0*e
    #settings.best_dt_array = Alaising(2,6,e)
    settings.best_dt_array = Alaising(1.2,99,BEST_ITER)
    settings.max_best_iter = 0

    # dropout parameter
    #settings.dropout_value = float(input("dropout value: "))
    settings.dropout_value = 0.05

    # weight_decay parameter
    #settings.weight_decay = float(input('weight_decay: '))
    settings.weight_decay = 0.0005
    # momentum parameter
    settings.momentum = 0.9

    # MAX LAYER MUTATION (CONDITION)
    MAX_LAYERS = 30

    # MAX FILTERS MUTATION (CONDITION)
    MAX_FILTERS = 256

    MAX_FILTERS_DENSE = 256

    list_conditions={DNA_conditions.max_filter : 530,
            DNA_conditions.max_filter_dense : 530,
            DNA_conditions.max_kernel_dense : 9,
            DNA_conditions.max_layer : 200,
            DNA_conditions.min_filter : 3,
            DNA_conditions.max_pool_layer : 10,
            DNA_conditions.max_parents : 2}

    # TEST_NAME, the name of the experiment (unique)
    settings.test_name = input("Enter TestName: ")

    # ENABLE_ACTIVATION, enable/disable relu
    ENABLE_ACTIVATION = int(input("Enable activation? (1 = yes, 0 = no): "))

    value = False
    if ENABLE_ACTIVATION == 1:
        value = True
    settings.enable_activation = value

    # ENABLE_LAST_ACTIVATION, enable/disable last layer relu
    ENABLE_LAST_ACTIVATION = int(input("Enable last layer activation? (1 = yes, 0 = no): "))

    value = False
    if ENABLE_LAST_ACTIVATION == 1:
        value = True
    settings.enable_last_activation = value

    # ENABLE_AUGMENTATION, enable/disable data augmentation
    ENABLE_AUGMENTATION = int(input("Enable Data augmentation? (1 = yes, 0 = no): "))

    value = False
    if ENABLE_AUGMENTATION == 1:
        value = True
    ENABLE_AUGMENTATION = value

    # ALLOW INTERRUPTS
    ALLOW_INTERRUPTS = int(input("Allow Interrupt while training? (1 = yes, 0 = no): "))
    value = False
    if ALLOW_INTERRUPTS == 1:
        value = True
    settings.allow_interupts = value

    # ALLOW TRACK BATCHNORM
    ENABLE_TRACK = int(input("Enable tracking var/mean batchnorm? (1 = yes, 0 = no): "))
    value = False
    if ENABLE_TRACK == 1:
        value = True
    settings.enable_track_stats = value

    # DROPOUT FUNCTION
    settings.dropout_function = dropout_function

    # INITIAL DNA
    DNA = ((-1, 1, 3, 32, 32), (0, 3, 128, 3, 3), (0, 128, 128, 3, 3), (0, 128, 128, 4, 4), 
                            (0, 128, 128, 3, 3), (0, 128, 256, 5, 5, 2), (0, 384, 128, 3, 3, 2), (0, 128, 128, 3, 3), 
                            (0, 128, 512, 3, 3, 2), (0, 128, 128, 2, 2), (0, 256, 256, 3, 3), (0, 256, 256, 5, 5), 
                            (0, 512, 256, 3, 3), (0, 256, 256, 3, 3), (0, 768, 512, 4, 4, 2), (0, 512, 512, 4, 4), 
                            (0, 512, 512, 2, 2), (0, 512, 512, 3, 3), (0, 1024, 256, 2, 2), (0, 256, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 0, 5), (3, 4, 5), (3, 5, 6), (3, 6, 7), (3, 5, 8), (3, 4, 9), (3, 8, 10), (3, 9, 10), (3, 10, 11), (3, 4, 11), (3, 11, 12), (3, 12, 13), (3, 7, 13), (3, 7, 14), (3, 14, 15), (3, 15, 16), (3, 13, 17), (3, 16, 17), (3, 17, 18), (3, 18, 19), (3, 19, 20))
    
    MUTATE_DNA = ((-1, 1, 3, 32, 32), (0, 3, 128, 3, 3), (0, 128, 128, 3, 3), (0, 128, 128, 4, 4), (0, 128, 128, 3, 3), 
                    (0, 128, 256, 5, 5, 2), (0, 384, 128, 3, 3, 2), (0, 128, 128, 3, 3), (0, 128, 512, 3, 3, 2),
                    (0, 128, 128, 2, 2), (0, 256, 256, 3, 3), (0, 256, 256, 5, 5), (0, 512, 256, 3, 3), (0, 256, 256, 3, 3), 
                    (0, 768, 512, 4, 4, 2), (0, 512, 512, 4, 4), (0, 512, 512, 2, 2), (0, 512, 512, 3, 3), 
                    (0, 1024, 512, 2, 2), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 0, 5), (3, 4, 5), (3, 5, 6), (3, 6, 7), (3, 5, 8), (3, 4, 9), (3, 8, 10), (3, 9, 10), (3, 10, 11), (3, 4, 11), (3, 11, 12), (3, 12, 13), (3, 7, 13), (3, 7, 14), (3, 14, 15), (3, 15, 16), (3, 13, 17), (3, 16, 17), (3, 17, 18), (3, 18, 19), (3, 19, 20))

    init_network = nw.Network(adn=DNA, cudaFlag=settings.cuda,
                                    momentum=settings.momentum, weight_decay=settings.weight_decay,
                                    enable_activation=settings.enable_activation,
                                    enable_track_stats=settings.enable_track_stats, dropout_value=settings.dropout_value,
                                    dropout_function=settings.dropout_function, 
                                    enable_last_activation=settings.enable_last_activation,
                                    version=settings.version, eps_batchnorm=settings.eps_batchorm)

    mutator = MutationManager.MutationManager(directions_version=settings.version)
    settings.loadedNetwork = mutator.executeMutation(init_network, MUTATE_DNA)

    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
    dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE, threads=THREADS, dataAugmentation=ENABLE_AUGMENTATION)
    dataGen = dataCreator.returnParam()

    space, selector = DNA_Creator_s(dataGen.size[1], dataGen.size[2], dna=settings.loadedNetwork.adn, version=settings.version)

    settings.dataGen = dataGen
    settings.selector = selector
    settings.initial_space = space
    settings.ricap = Augmentation.Ricap(beta=0.3)

    print('The initial DNA is:')
    print(settings.loadedNetwork.adn)

    trainer = CommandExperimentCifar_Restarts.CommandExperimentCifar_Restarts(settings=settings)
    trainer.execute()
