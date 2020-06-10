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
    max_layers=10
    max_filters=60
    max_dense=100
    def condition_b(z):
        return max_filter_dense(max_filter(max_layer(z,max_layers),max_filters),max_dense)

    center=dna
    mutations=((4,0,0,0),(1,0,0,0),(0,0,1))
    sel=Selector_creator(condition=condition_b,
        directions=version,mutations=mutations,num_actions=num_actions)
    print('The selector is')
    print(sel)
    sel.update(center)
    actions=sel.get_predicted_actions()
    creator=Creator_nm
    space=DNA_Graph(center,1,(x,y),condition_b,actions,
        version,creator=creator,num_morphisms=5,selector=sel)
    
    return [space, sel]

if __name__ == '__main__':

    settings = ExperimentSettings.ExperimentSettings()
    
    random_erase_1 = int(input("RandomErase (size: 4x4, holes: 2) ? (1 -> yes, 0 -> no): "))
    random_erase_2 = int(input("RandomErase (size: 2x2, holes: 16) ? (1 -> yes, 0 -> no): "))

    enable_randomerase_1 = False
    enable_randomerase_2 = False
    
    if random_erase_1 == 1:
        enable_randomerase_1 = True
    
    if random_erase_2 == 1:
        enable_randomerase_2 = True

    augSettings = AugmentationSettings.AugmentationSettings()

    dict_transformations = {
        augSettings.baseline_customRandomCrop : True,
        augSettings.cutout : True,
        augSettings.randomErase_1 : enable_randomerase_1,
        augSettings.randomErase_2 : enable_randomerase_2,
        augSettings.translate : True,
        augSettings.randomHorizontalFlip : True,
        augSettings.randomShear: True,
    }

    transform_compose = augSettings.generateTransformCompose(dict_transformations, False)

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

    num_actions=5
    e=10

    # INITIAL DT PARAMETERS
    max_init_iter = 1
    settings.max_init_iter = max_init_iter
    INIT_ITER = 20*e
    #settings.init_dt_array = exp_alai(.5,INIT_ITER,1,5)
    settings.init_dt_array =  Alaising(1,7,INIT_ITER)


    # JOINED DT PARAMETERS
    JOINED_ITER = 4*e
    #settings.joined_dt_array = Alaising(2,6,e)
    settings.joined_dt_array = Alaising(1.2,7,JOINED_ITER)
    settings.max_joined_iter = 1

    # BEST DT PARAMETERS
    BEST_ITER = 10*e
    #settings.best_dt_array = Alaising(2,6,e)
    settings.best_dt_array = Alaising(1.2,7,BEST_ITER)
    settings.max_best_iter = 1

    # dropout parameter
    settings.dropout_value = float(input("dropout value: "))

    # weight_decay parameter
    settings.weight_decay = float(input('weight_decay: '))

    # momentum parameter
    settings.momentum = 0.9

    list_conditions={DNA_conditions.max_filter : 530,
            DNA_conditions.max_filter_dense : 530,
            DNA_conditions.max_kernel_dense : 1,
            DNA_conditions.max_layer : 30,
            DNA_conditions.min_filter : 0,
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
    settings.initial_dna =   ((-1,1,3,32,32),(0,3, 5, 3 , 3),(0,5, 6, 3,  3), (0,6,7,3,3,2),
            (0,7, 8, 16,16),(1, 8,10),(2,),(3,-1,0), (3,0,1),(3,1,2),(3,2,3),(3,3,4),(3,4,5))


    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
    dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE, threads=THREADS, dataAugmentation=ENABLE_AUGMENTATION, transformCompose=transform_compose)
    dataGen = dataCreator.returnParam()

    space, selector = DNA_Creator_s(dataGen.size[1], dataGen.size[2], dna=settings.initial_dna, version=settings.version)

    settings.dataGen = dataGen
    settings.selector = selector
    settings.initial_space = space

    settings.save_txt = False

    settings.disable_mutation = False

    print("**** WARNING DISABLE MUTATION = ", settings.disable_mutation)
    trainer = CommandExperimentCifar_Restarts.CommandExperimentCifar_Restarts(settings=settings)
    trainer.execute()
