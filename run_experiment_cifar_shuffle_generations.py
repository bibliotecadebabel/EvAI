from TestNetwork.commands import CommandCreateDataGen
from TestNetwork.commands import  CommandExperimentCifar_Shuffle_generations as CommandExperimentCifar_Restarts
from DNA_conditions import max_layer,max_filter,max_filter_dense
from DNA_creators import Creator
from DNA_Graph import DNA_Graph
from DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.positive_random_selector import centered_random_selector as random_selector
import TestNetwork.ExperimentSettings as ExperimentSettings
import numpy as np
###### EXPERIMENT SETTINGS ######

def dropout_function(base_p, total_conv2d, index_conv2d):

    value = base_p / (total_conv2d - index_conv2d)+0.1
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

def DNA_Creator_s(x,y, dna):
    def condition(DNA):
        return max_filter_dense(max_filter(max_layer(DNA, MAX_LAYERS), MAX_FILTERS), MAX_FILTERS_DENSE)

    version='pool'
    selector = None
    selector=random_selector(condition=condition,
        directions=version, num_actions=num_actions)
    selector.update(dna)
    actions=selector.get_predicted_actions()
    #actions = ((0, (0,1,0,0)), (1, (0,1,0,0)), (0, (1,0,0,0)))
    space=DNA_Graph(dna,1,(x,y),condition,actions
        ,version,Creator_s)

    return [space, selector]

if __name__ == '__main__':

    settings = ExperimentSettings.ExperimentSettings()

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

    e=300
    settings.max_init_iter = 2
    INIT_ITER = 20*e
    #settings.init_dt_array = exp_alai(.5,INIT_ITER,1,5)
    settings.init_dt_array =  Alaising(1,5,INIT_ITER)


    # JOINED DT PARAMETERS
    JOINED_ITER = 4*e
    #settings.joined_dt_array = Alaising(2,6,e)
    settings.joined_dt_array = Alaising(2,5,JOINED_ITER)
    settings.max_joined_iter = 1

    # BEST DT PARAMETERS
    BEST_ITER = 7*e
    #settings.best_dt_array = Alaising(2,6,e)
    settings.best_dt_array = Alaising(2,5,BEST_ITER)
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
    MAX_FILTERS = 65

    MAX_FILTERS_DENSE = 130

    # TEST_NAME, the name of the experiment (unique)
    settings.test_name = input("Enter TestName: ")

    # ENABLE_ACTIVATION, enable/disable sigmoid + relu
    ENABLE_ACTIVATION = int(input("Enable activation? (1 = yes, 0 = no): "))

    value = False
    if ENABLE_ACTIVATION == 1:
        value = True
    settings.enable_activation = value

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
    settings.initial_dna =   ((-1, 1, 3, 32, 32), (0, 3, 16, 3, 3),(0, 16, 32, 3, 3, 2), (0, 32, 64, 3, 3, 2),
                                (0, 64, 128, 5, 5),
                                (1, 128, 10),
                                (2,),
                                (3, -1, 0),
                                (3, 0, 1),
                                (3, 1, 2),
                                (3, 2, 3),
                                (3, 3, 4),
                                (3, 4, 5))

    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
    dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE, threads=THREADS, dataAugmentation=ENABLE_AUGMENTATION)
    dataGen = dataCreator.returnParam()

    space, selector = DNA_Creator_s(dataGen.size[1], dataGen.size[2], dna=settings.initial_dna)

    settings.dataGen = dataGen
    settings.selector = selector
    settings.initial_space = space

    trainer = CommandExperimentCifar_Restarts.CommandExperimentCifar_Restarts(settings=settings)
    trainer.execute()
