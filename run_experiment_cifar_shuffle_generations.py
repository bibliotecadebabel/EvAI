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
###### EXPERIMENT SETTINGS ######
"""
def dropout_function(base_p, total_conv2d, index_conv2d):

    value = base_p / (total_conv2d - index_conv2d)+base_p/2
    #print("conv2d: ", index_conv2d, " - dropout: ", value)
    return value


"""
"""
def dropout_function(base_p, total_conv2d, index_conv2d):
    value = base_p +(3/5*base_p-base_p)*(total_conv2d - index_conv2d)/total_conv2d
    #print("conv2d: ", index_conv2d, " - dropout: ", value)
    return value
"""

def dropout_function(base_p, total_layers, index_layer, isPool=False):

    value = 0
    if index_layer != 0 and isPool == False:
        value = base_p +(3/5*base_p-base_p)*(total_layers - index_layer-1)/total_layers

    if index_layer == total_layers - 2:
        value = base_p +(3/5*base_p-base_p)*(total_layers - index_layer-1)/total_layers

    #print("conv2d: ", index_layer, " - dropout: ", value)

    return value




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

    # DIRECTIONS VERSION
    settings.version = directions_version.H_VERSION
    # NUM OF THREADS
    THREADS = int(input("Enter threads: "))

    # BATCH SIZE
    settings.batch_size = int(input("Enter batchsize: "))

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
    num_actions=5

    e=400
    settings.max_init_iter = 150
    INIT_ITER = 10*e
    #settings.init_dt_array = exp_alai(.5,INIT_ITER,1,5)
    settings.init_dt_array =  Alaising(1.2,7,INIT_ITER)


    # JOINED DT PARAMETERS
    JOINED_ITER = 0
    #settings.joined_dt_array = Alaising(2,6,e)
    settings.joined_dt_array = Alaising(1.2,5,JOINED_ITER)
    settings.max_joined_iter = 1

    # BEST DT PARAMETERS
    BEST_ITER = 0
    #settings.best_dt_array = Alaising(2,6,e)
    settings.best_dt_array = Alaising(1.2,7,BEST_ITER)
    settings.max_best_iter = 1

    # dropout parameter
    settings.dropout_value = float(input("dropout value: "))

    # weight_decay parameter
    settings.weight_decay = float(input('weight_decay: '))

    # momentum parameter
    settings.momentum = 0.9

    # MAX LAYER MUTATION (CONDITION)
    MAX_LAYERS = 30

    # MAX FILTERS MUTATION (CONDITION)
    MAX_FILTERS = 256

    MAX_FILTERS_DENSE = 256

    list_conditions={DNA_conditions.max_filter : 257,
            DNA_conditions.max_filter_dense : 257,
            DNA_conditions.max_kernel_dense : 9,
            DNA_conditions.max_layer : 200,
            DNA_conditions.min_filter : 3,
            DNA_conditions.max_pool_layer : 4,
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

    """
    settings.initial_dna = ((-1, 1, 3, 32, 32), (0, 3, 6, 3, 3),
                                (0, 9, 12, 3, 3), (0, 12, 24, 3, 3, 2),
                                 (0, 36, 24, 3, 3, 2), (0, 39, 6, 3, 3),
                                 (0, 6, 6, 2, 2, 2), (0, 15, 6, 2, 2, 2),
                                 (0, 12, 6, 2, 2, 2), (0, 12, 64, 3, 3),
                                 (0, 73, 64, 3, 3), (0, 70, 32, 2, 2),
                                 (0, 32, 32, 3, 3), (0, 131, 32, 4, 4),
                                  (0, 32, 32, 2, 2, 2), (0, 64, 8, 4, 4),
                                   (0, 75, 8, 3, 3, 2), (0, 8, 16, 4, 4, 2),
                                    (0, 19, 16, 2, 2, 2), (0, 40, 16, 2, 2, 2),
                                     (0, 16, 16, 2, 2, 2), (0, 88, 128, 3, 3, 2),
                                      (0, 128, 128, 5, 5), (0, 128, 64, 2, 2, 2),
                                       (0, 320, 128, 2, 2), (0, 160, 128, 2, 2, 2),
                                        (0, 256, 128, 3, 3), (0, 256, 64, 2, 2, 2),
                                         (0, 64, 128, 4, 4), (0, 384, 128, 3, 3, 2),
                                         (0, 384, 128, 12, 12), (1, 128, 10),
                                          (2,), (3, -1, 0), (3, -1, 1), (3, 0, 1), (3, 1, 2), (3, 1, 3), (3, 2, 3), (3, -1, 4), (3, 1, 4), (3, 3, 4), (3, 4, 5),
                                          (3, -1, 6), (3, 4, 6), (3, 5, 6), (3, 6, 7), (3, 4, 7), (3, 6, 8), (3, 7, 8), (3, -1, 9), (3, 6, 9), (3, 8, 9), (3, 9, 10),
                                           (3, 6, 10), (3, 10, 11), (3, -1, 12), (3, 10, 12), (3, 9, 12), (3, 11, 12), (3, 12, 13), (3, 12, 14), (3, 13, 14), (3, 12, 15),
                                            (3, -1, 15), (3, 14, 15), (3, 10, 15), (3, 15, 16), (3, 16, 17), (3, -1, 17), (3, 15, 18), (3, 17, 18), (3, 16, 18), (3, 18, 19),
                                            (3, 12, 20), (3, 15, 20), (3, 18, 20), (3, 16, 20), (3, 19, 20), (3, 20, 21), (3, 21, 22), (3, 20, 23), (3, 21, 23), (3, 22, 23),
                                            (3, 23, 24), (3, 12, 24), (3, 23, 25), (3, 24, 25), (3, 25, 26), (3, 24, 26), (3, 26, 27), (3, 20, 28), (3, 23, 28), (3, 27, 28),
                                             (3, 28, 29), (3, 23, 29), (3, 27, 29), (3, 29, 30), (3, 30, 31))

    """

    settings.initial_dna =   DNAs.DNA_ep25


    """
    settings.initial_dna =   ((-1, 1, 3, 32, 32), (0, 3, 32, 3, 3),(0, 32, 64, 3, 3, 2), (0, 64, 128, 3, 3, 2),
                                (0, 128, 256, 8, 8),
                                (1, 256, 10),
                                (2,),
                                (3, -1, 0),
                                (3, 0, 1),
                                (3, 1, 2),
                                (3, 2, 3),
                                (3, 3, 4),
                                (3, 4, 5))
    """
    """

    settings.initial_dna =   ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3),(0, 64, 128, 3, 3, 2),
                                (0, 128, 256, 4, 4,4),
                                (1, 256, 10),
                                (2,),
                                (3, -1, 0),
                                (3, 0, 1),
                                (3, 1, 2),
                                (3, 2, 3),
                                (3, 3, 4))
    """


    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
    dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE, threads=THREADS, dataAugmentation=ENABLE_AUGMENTATION)
    dataGen = dataCreator.returnParam()

    space, selector = DNA_Creator_s(dataGen.size[1], dataGen.size[2], dna=settings.initial_dna, version=settings.version)

    settings.dataGen = dataGen
    settings.selector = selector
    settings.initial_space = space

    trainer = CommandExperimentCifar_Restarts.CommandExperimentCifar_Restarts(settings=settings)
    trainer.execute()
