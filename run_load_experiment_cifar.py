from TestNetwork.commands import CommandCreateDataGen
from TestNetwork.commands import  CommandExperimentCifar_Shuffle_generations as CommandExperimentCifar_Restarts
from DNA_conditions import max_layer,max_filter,max_filter_dense
import DNA_conditions
from DNA_creators import Creator
from DNA_Graph import DNA_Graph
from DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.uniform_random_selector import centered_random_selector as random_selector
import TestNetwork.ExperimentSettings as ExperimentSettings
import TestNetwork.AugmentationSettings as AugmentationSettings
import const.versions as directions_version
import numpy as np
import test_DNAs as DNAs
import utilities.NetworkStorage as NetworkStorage

###### EXPERIMENT SETTINGS ######
"""
def dropout_function(base_p, total_conv2d, index_conv2d):

    value = base_p / (total_conv2d - index_conv2d)+base_p/2
    #print("conv2d: ", index_conv2d, " - dropout: ", value)
    return value
"""

def dropout_function(base_p, total_conv2d, index_conv2d):
    value = 3/5*base_p +(base_p-3/5*base_p)*1/ (total_conv2d - index_conv2d)
    print("conv2d: ", index_conv2d, " - dropout: ", value)
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
        (4,0,0,0),
        (1,0,0,0),
        (0,0,1),
        ))
    selector.update(dna)
    actions=selector.get_predicted_actions()
    #actions = ((0, (0,1,0,0)), (1, (0,1,0,0)), (0, (1,0,0,0)))
    space=DNA_Graph(dna,1,(x,y),condition,actions
        ,version,Creator_s)

    return [space, selector]

if __name__ == '__main__':

    settings = ExperimentSettings.ExperimentSettings()
    augmentation_settings = AugmentationSettings.AugmentationSettings()

    # MODEL NAME
    model_name = "test_red_storage_1"
    
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

    # INITIAL DT PARAMETERS
    num_actions=5

    e=10
    settings.max_init_iter = 40
    INIT_ITER = 20*e
    #settings.init_dt_array = exp_alai(.5,INIT_ITER,1,5)
    settings.init_dt_array =  Alaising(1,5,INIT_ITER)


    # JOINED DT PARAMETERS
    JOINED_ITER = 4*e
    #settings.joined_dt_array = Alaising(2,6,e)
    settings.joined_dt_array = Alaising(1.2,5,JOINED_ITER)
    settings.max_joined_iter = 1

    # BEST DT PARAMETERS
    BEST_ITER = 7*e
    #settings.best_dt_array = Alaising(2,6,e)
    settings.best_dt_array = Alaising(1.2,5,BEST_ITER)
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
    MAX_FILTERS = 130

    MAX_FILTERS_DENSE = 130

    list_conditions={DNA_conditions.max_filter : 520,
            DNA_conditions.max_filter_dense : 520,
            DNA_conditions.max_kernel_dense : 9,
            DNA_conditions.max_layer : 30,
            DNA_conditions.min_filter : 0,
            DNA_conditions.max_parents : 2}
    
    '''
    transform_list = { 
        ## NEW 
        augmentation_settings.contrast : False,
        augmentation_settings.zoomout : False,
        
        ## DEFAULT
        augmentation_settings.randomAffine : False,
        augmentation_settings.randomHorizontalFlip : False       
    }
    '''

    # ENABLE OR DISABLE FIVE CROP.
    ENABLE_FIVE_CROP = True

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


    #TRANSFORM_COMPOSE = augmentation_settings.generateTransformCompose(transform_list, ENABLE_FIVE_CROP)
    TRANSFORM_COMPOSE = None
    settings.loadedNetwork = NetworkStorage.loadNetwork(fileName=model_name, settings=settings)
    
    settings.version = settings.loadedNetwork.version
    settings.dropout_function = dropout_function

    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
    dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE, threads=THREADS, 
                            dataAugmentation=ENABLE_AUGMENTATION, transformCompose=TRANSFORM_COMPOSE)
    dataGen = dataCreator.returnParam()

    space, selector = DNA_Creator_s(dataGen.size[1], dataGen.size[2], dna=settings.loadedNetwork.adn, version=settings.version)

    settings.dataGen = dataGen
    settings.selector = selector
    settings.initial_space = space

    trainer = CommandExperimentCifar_Restarts.CommandExperimentCifar_Restarts(settings=settings)
    trainer.execute()
