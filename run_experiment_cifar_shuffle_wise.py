from TestNetwork.commands import CommandCreateDataGen
from TestNetwork.commands import  CommandExperimentCifar_Shuffle_ver_2 as CommandExperimentCifar_Restarts
from DNA_conditions import max_layer,max_filter
from DNA_creators import Creator
from DNA_Graph import DNA_Graph
from DNA_creator_duplicate_clone import Creator_from_selection_clone as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector
import TestNetwork.ExperimentSettings as ExperimentSettings

###### EXPERIMENT SETTINGS ######


settings = ExperimentSettings.ExperimentSettings()

# BATCH SIZE
settings.batch_size = int(input("Enter batchsize: "))

# DATA SOURCE ('default' -> Pikachu, 'cifar' -> CIFAR)
DATA_SOURCE = 'cifar'

# Every PERDIOD_SAVE iterations, all DNA and its energies will be stored in the database.
settings.period_save_space = 1

# Every PERDIOD_NEWSPACE iterations, a new DNA GRAPH will be generated with the dna of the lowest energy network as center.
settings.period_new_space = 1

# Every PERIOD_SAVE_MODEL iterations, the best network (current center) will be stored on filesystem
settings.period_save_model = 1

# EPOCHS
settings.epochs = int(input("Enter amount of epochs: "))

# INITIAL DT PARAMETERS
settings.max_init_iter = 4
INIT_ITER = 7*300
settings.init_dt_array = [ 10 ** (-1-k/INIT_ITER) for k in range(INIT_ITER)]


# JOINED DT PARAMETERS
JOINED_ITER = 600
settings.joined_dt_array = [ 10 ** (-3-(k/JOINED_ITER)) for k in range(JOINED_ITER)]
settings.max_joined_iter = 2

# BEST DT PARAMETERS
BEST_ITER = 2000
settings.best_dt_array = [ 10 ** (-3-(k/BEST_ITER)) for k in range(BEST_ITER)]
settings.max_best_iter = 4

# weight_decay parameter
settings.weight_decay = 0.00001

# momentum parameter
settings.momentum = 0.9

# CUDA parameter (true/false)
settings.cuda = True

# MAX LAYER MUTATION (CONDITION)
MAX_LAYERS = 15

# MAX FILTERS MUTATION (CONDITION)
MAX_FILTERS = 42

# TEST_NAME, the name of the experiment (unique)
settings.test_name = input("Enter TestName: ")

# ENABLE_ACTIVATION, enable/disable sigmoid + relu
ENABLE_ACTIVATION = int(input("Enable activation? (1 = yes, 0 = no): "))

value = False
if ENABLE_ACTIVATION == 1:
    value = True
settings.enable_activation = value

# INITIAL DNA
settings.initial_dna = ((-1,1,3,32,32),
        (0,3, 5, 3 , 3),
        (0,8, 5, 3,  3),
        (0,13, 41, 32, 32),
        (1, 41,10),
         (2,),
        (3,-1,0),
        (3,0,1),(3,-1,1),
        (3,1,2),(3,0,2),(3,-1,2),
        (3,2,3),
        (3,3,4))


def DNA_Creator_s(x,y, dna):
    def condition(DNA):
        return max_filter(max_layer(DNA,MAX_LAYERS),MAX_FILTERS)

    selector = None
    selector=random_selector()
    selector.update(dna)
    actions=selector.get_predicted_actions()
    version='final'
    #actions = ((0, (0,1,0,0)), (1, (0,1,0,0)), (0, (1,0,0,0)))
    space=DNA_Graph(dna,1,(x,y),condition,actions
        ,version,Creator_s)

    return [space, selector]

dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE)
dataGen = dataCreator.returnParam()

space, selector = DNA_Creator_s(dataGen.size[1], dataGen.size[2], dna=settings.initial_dna)

settings.dataGen = dataGen
settings.selector = selector
settings.initial_space = space

trainer = CommandExperimentCifar_Restarts.CommandExperimentCifar_Restarts(settings=settings)
trainer.execute()
