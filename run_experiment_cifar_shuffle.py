from TestNetwork.commands import CommandCreateDataGen
from TestNetwork.commands import  CommandExperimentCifar_Shuffle as CommandExperimentCifar_Restarts
from DNA_conditions import max_layer,max_filter
from DNA_creators import Creator
from DNA_Graph import DNA_Graph
from DNA_creator_duplicate_clone import Creator_from_selection_clone as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector

###### EXPERIMENT SETTINGS ######



# BATCH SIZE
BATCH_SIZE = int(input("Enter batchsize: "))

# DATA SOURCE ('default' -> Pikachu, 'cifar' -> CIFAR)
DATA_SOURCE = 'cifar'

# Every PERDIOD_SAVE iterations, all DNA and its energies will be stored in the database.
PERIOD_SAVE = 1

# Every PERDIOD_NEWSPACE iterations, a new DNA GRAPH will be generated with the dna of the lowest energy network as center.
PERIOD_NEWSPACE = 1

# Every PERIOD_SAVE_MODEL iterations, the best network (current center) will be stored on filesystem
PERIOD_SAVE_MODEL = 1

# EPOCHS
EPOCHS = int(input("Enter amount of epochs: "))

# DT ARRAY #1
DT_ARRAY_1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# DT ARRAY #2
DT_ARRAY_2 = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

# weight_decay parameter
WEIGHT_DECAY = 0.00001

# momentum parameter
MOMENTUM = 0.9

# CUDA parameter (true/false)
CUDA = True

# MAX LAYER MUTATION (CONDITION)
MAX_LAYERS = 15

# MAX FILTERS MUTATION (CONDITION)
MAX_FILTERS = 240

# TEST_NAME, the name of the experiment (unique)
TEST_NAME = input("Enter TestName: ")

# ENABLE_ACTIVATION, enable/disable sigmoid + relu
ENABLE_ACTIVATION = int(input("Enable activation? (1 = yes, 0 = no): "))

value = False
if ENABLE_ACTIVATION == 1:
    value = True
ENABLE_ACTIVATION = value

# INITIAL DNA
DNA = ((-1,1,3,32,32),
        (0,3, 5, 3 , 3),
        (0,5, 5, 3,  3),
        (0,5, 120, 32-4, 32-4),
        (1, 120,10),
        (2,),
        (3,-1,0),
        (3,0,1),
        (3,1,2),
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


if CUDA:

    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=CUDA)



dataCreator.execute(compression=2, batchSize=BATCH_SIZE, source=DATA_SOURCE)
dataGen = dataCreator.returnParam()

space, selector = DNA_Creator_s(dataGen.size[1], dataGen.size[2], dna=DNA)

trainer = CommandExperimentCifar_Restarts.CommandExperimentCifar_Restarts(initialDNA=DNA, dataGen=dataGen, testName=TEST_NAME,
                                                                            selector=selector, weight_decay=WEIGHT_DECAY,
                                                                                momentum=MOMENTUM, space=space, cuda=CUDA,
                                                                                enable_activation=ENABLE_ACTIVATION)

trainer.execute(periodSave=PERIOD_SAVE, periodNewSpace=PERIOD_NEWSPACE, periodSaveModel=PERIOD_SAVE_MODEL, epochs=EPOCHS,
                dt_array_1=DT_ARRAY_1, dt_array_2=DT_ARRAY_2)
