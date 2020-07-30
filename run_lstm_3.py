from Logic.commands import CommandCreateDataGen
from Logic.commands import  CommandExperiment_lstm as CommandExperimentCifar_Restarts
from Geometric.Conditions.DNA_conditions import max_layer,max_filter,max_filter_dense
import Geometric.Conditions.DNA_conditions as DNA_conditions
from Geometric.Creators.DNA_creators import Creator
from Geometric.Graphs.DNA_Graph import DNA_Graph
from Geometric.Creators.DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.lstm_selector import centered_random_selector as random_selector
import utilities.ExperimentSettings as ExperimentSettings
import const.versions as directions_version
import numpy as np
import tests_scripts.test_DNAs as DNAs
import utilities.Augmentation as Augmentation
import utilities.AugmentationSettings as AugmentationSettings
from utilities.Abstract_classes.classes.Alaising_cosine import (
    Damped_Alaising as Alai_w)
import math
###### EXPERIMENT SETTINGS ######

def dropout_function(base_p, total_layers, index_layer, isPool=False):

    value = 0
    if index_layer != 0 and isPool == False:
        value = base_p

    if index_layer == total_layers - 2:
        value = base_p

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
        mutations=((1,0,0,0),(4,0,0,0),(0,1,0,0),(0,-1,0,0),(0,0,1,1),(0,0,-1,-1),(0,0,1),(0,0,-1)) 
    )
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
    settings.version = directions_version.POOL_VERSION_DUPLICATE
    # NUM OF THREADS
    THREADS = int(input("Enter threads: "))

    # BATCH SIZE
    settings.batch_size = 64

    e =  50000 / settings.batch_size
    e = math.ceil(e)
    print("e = ", e)
    settings.iteration_per_epoch = e

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
    settings.epochs = 2000

    settings.eps_batchorm = 0.001

    # INITIAL DT PARAMETERS
    num_actions=8
    settings.save_txt = True
    settings.max_init_iter = 0
    INIT_ITER = 200*e
    #alai_w=Alai_w(initial_max=0.1,final_max=0.05,
    #    initial_min=10**(-99),final_min=10**(-99),Max_iter=INIT_ITER)
    #settings.init_dt_array = alai_w.get_increments(225*e)
    #settings.init_dt_array = exp_alai(1,INIT_ITER,1,7)
    settings.init_dt_array =  Alaising(1.2,99,INIT_ITER)


    # JOINED DT PARAMETERS
    JOINED_ITER = 100*e
    #settings.joined_dt_array = Alaising(2,6,e)
    settings.joined_dt_array = Alaising(1.2,99,JOINED_ITER)
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
            DNA_conditions.max_filter_dense : 270,
            DNA_conditions.max_kernel_dense : 9,
            DNA_conditions.max_layer : 20,
            DNA_conditions.min_filter : 3,
            DNA_conditions.max_pool_layer : 4,
            DNA_conditions.max_parents : 2,
            DNA_conditions.no_con_last_layer : 1}

    # TEST_NAME, the name of the experiment (unique)
    settings.test_name = "test_lstm_3"

    # ENABLE_ACTIVATION, enable/disable relu
    ENABLE_ACTIVATION = 1

    value = False
    if ENABLE_ACTIVATION == 1:
        value = True
    settings.enable_activation = value

    # ENABLE_LAST_ACTIVATION, enable/disable last layer relu
    ENABLE_LAST_ACTIVATION = 1

    value = False
    if ENABLE_LAST_ACTIVATION == 1:
        value = True
    settings.enable_last_activation = value

    # ENABLE_AUGMENTATION, enable/disable data augmentation
    ENABLE_AUGMENTATION = 1

    value = False
    if ENABLE_AUGMENTATION == 1:
        value = True
    ENABLE_AUGMENTATION = value

    # ALLOW INTERRUPTS
    ALLOW_INTERRUPTS = 0
    value = False
    if ALLOW_INTERRUPTS == 1:
        value = True
    settings.allow_interupts = value

    # ALLOW TRACK BATCHNORM
    ENABLE_TRACK = 1
    value = False
    if ENABLE_TRACK == 1:
        value = True
    settings.enable_track_stats = value

    # DROPOUT FUNCTION

    settings.dropout_function = dropout_function
    # INITIAL DNA

    settings.initial_dna = DNAs.DNA_base_p_version

    print('The initial DNA is:')
    print(settings.initial_dna)


    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
    dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE, threads=THREADS, dataAugmentation=ENABLE_AUGMENTATION)
    dataGen = dataCreator.returnParam()

    space, selector = DNA_Creator_s(dataGen.size[1], dataGen.size[2], dna=settings.initial_dna, version=settings.version)

    settings.dataGen = dataGen
    settings.selector = selector
    settings.initial_space = space
    settings.ricap = Augmentation.Ricap(beta=0.3)

    trainer = CommandExperimentCifar_Restarts.CommandExperimentCifar_Restarts(settings=settings)
    trainer.execute()
