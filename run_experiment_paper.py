from TestNetwork.commands import CommandCreateDataGen
from TestNetwork.commands import  CommandExperiment_Paper as CommandExperimentCifar_Restarts
import DNA_conditions
from DNA_Graph import DNA_Graph
import TestNetwork.ExperimentSettings as ExperimentSettings
import TestNetwork.AugmentationSettings as AugmentationSettings
import const.versions as directions_version
import numpy as np
import test_DNAs as DNAs

from utilities.Abstract_classes.classes.uniform_random_selector_2 import(
    centered_random_selector as Selector_creator)
from DNA_conditions import max_layer,max_filter,max_filter_dense
from DNA_creators import Creator_from_selection_nm as Creator_nm
import utilities.Augmentation as Augmentation_Utils
import math

###### EXPERIMENT SETTINGS ######
"""
def dropout_function(base_p, total_conv2d, index_conv2d):

    value = base_p / (total_conv2d - index_conv2d)+base_p/2
    #print("conv2d: ", index_conv2d, " - dropout: ", value)
    return value


"""

def dropout_function(base_p, total_layers, index_layer, isPool=False):

    value = 0
    if index_layer != 0 and isPool == False:
        value = base_p

    if index_layer == total_layers - 2:
        value = base_p

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

    center=dna
    mutations=(
    (1,0,0,0),(1,0,0,0),
    (0,1,0,0),(0,1,0,0),
    (4,0,0,0),
    (0,0,1),(0,0,-1),
    (0,0,1,1),(0,0,-1,-1),
    (0,0,2)
    )
    sel=Selector_creator(condition=condition,
        directions=version,mutations=mutations,num_actions=num_actions)
    print('The selector is')
    print(sel)
    sel.update(center)
    actions=sel.get_predicted_actions()
    creator=Creator_nm
    space=DNA_Graph(center,1,(x,y),condition,actions,version,creator=creator,num_morphisms=1,selector=sel)
    
    return [space, sel]

if __name__ == '__main__':

    settings = ExperimentSettings.ExperimentSettings()

    augSettings = AugmentationSettings.AugmentationSettings()

    dict_transformations = {
        augSettings.baseline_customRandomCrop : True,
        augSettings.randomHorizontalFlip : True,
        augSettings.randomErase_1 : True
    }

    transform_compose = augSettings.generateTransformCompose(dict_transformations, False)

    # DIRECTIONS VERSION
    settings.version = directions_version.CONVEX_VERSION
    # NUM OF THREADS
    THREADS = int(input("Enter threads: "))

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

    num_actions=1

    settings.batch_size = int(input("Enter batchsize: "))
    e =  50000 / settings.batch_size
    e = math.ceil(e)
    print("e = ", e)

    # INITIAL DT PARAMETERS
    max_init_iter = 0
    settings.max_init_iter = max_init_iter
    INIT_ITER = 20*e
    #settings.init_dt_array = exp_alai(.5,INIT_ITER,1,5)
    settings.init_dt_array =  Alaising(1,7,INIT_ITER)


    # JOINED DT PARAMETERS
    JOINED_ITER = 17*e
    #settings.joined_dt_array = Alaising(2,6,e)
    settings.joined_dt_array = Alaising(1.2,7,JOINED_ITER)
    settings.max_joined_iter = 0

    # BEST DT PARAMETERS
    BEST_ITER = 10*e
    #settings.best_dt_array = Alaising(2,6,e)
    settings.best_dt_array = Alaising(1.2,7,BEST_ITER)
    settings.max_best_iter = 0

    # dropout parameter
    settings.dropout_value = float(input("dropout value: "))

    # weight_decay parameter
    settings.weight_decay = float(input('weight_decay: '))

    # momentum parameter
    settings.momentum = 0.9

    # INITIAL DNA
    settings.initial_dna =  ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3),(0, 64, 128, 3, 3, 2), (0, 128, 256, 3, 3, 2),
                            (0, 256, 128, 8, 8),
                            (1, 128, 10),
                            (2,),
                            (3, -1, 0),
                            (3, 0, 1),
                            (3, 1, 2),
                            (3, 2, 3),
                            (3, 3, 4),
                            (3, 4, 5))
                            
    list_conditions={DNA_conditions.max_filter : 530,
            DNA_conditions.max_filter_dense : 260,
            DNA_conditions.max_kernel_dense : 9,
            DNA_conditions.max_layer : 200,
            DNA_conditions.min_filter : 3,
            DNA_conditions.max_parents : 2,
            DNA_conditions.no_con_last_layer : 1}

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
    settings.ricap = Augmentation_Utils.Ricap(beta=0.3)

    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
    dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE, threads=THREADS, dataAugmentation=ENABLE_AUGMENTATION, transformCompose=transform_compose)
    dataGen = dataCreator.returnParam()

    space, selector = DNA_Creator_s(dataGen.size[1], dataGen.size[2], dna=settings.initial_dna, version=settings.version)

    settings.dataGen = dataGen
    settings.selector = selector
    settings.initial_space = space

    settings.save_txt = True

    settings.disable_mutation = False

    print("**** WARNING DISABLE MUTATION = ", settings.disable_mutation)
    trainer = CommandExperimentCifar_Restarts.CommandExperimentCifar_Restarts(settings=settings)
    trainer.execute()
