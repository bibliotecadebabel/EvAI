from TestNetwork.commands import CommandCreateDataGen, CommandExperimentCifar_Restarts_monotone
from TestNetwork.commands import  CommandExperimentCifar_Restarts_monotone as CommandExperimentCifar_Restarts
from DNA_conditions import max_layer,max_filter
from DNA_creators import Creator
from DNA_Graph import DNA_Graph
from DNA_creator_duplicate_clone import Creator_from_selection_clone as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector

###### EXPERIMENT SETTINGS ######

# BATCH SIZE
BATCH_SIZE = 128

# DATA SOURCE ('default' -> Pikachu, 'cifar' -> CIFAR)
DATA_SOURCE = 'cifar'

# Every PERDIOD_SAVE iterations, all DNA and its energies will be stored in the database.
PERIOD_SAVE = 1

# Every PERDIOD_NEWSPACE iterations, a new DNA GRAPH will be generated with the dna of the lowest energy network as center.
PERIOD_NEWSPACE = 1

# Every PERIOD_SAVE_MODEL iterations, the best network (current center) will be stored on filesystem
PERIOD_SAVE_MODEL = 1

# EPOCHS
EPOCHS = 200

# MAX - MIN DTs FOR EPOCHS 1 TO 10
MAX_DT = 0.08
MIN_DT = 0.00001

# MAX - MIN DTs FOR THE REST OF THE EXPERIMENT
MAX_DT_2 = 0.001
MIN_DT_2 = 0.000001

# weight_decay parameter
WEIGHT_DECAY = 0.00001

# momentum parameter
MOMENTUM = 0.9

# CUDA parameter (true/false)
CUDA = True

# MAX LAYER MUTATION (CONDITION)
MAX_LAYERS = 15

TRIAL_EPOCS=3
# MAX FILTERS MUTATION (CONDITION)
MAX_FILTERS = 61

# TEST_NAME, the name of the experiment (unique)
TEST_NAME = "cifar_experiment_ver2"

# INITIAL DNA
DNA = ((-1,1,3,32,32),
        (0,3, 5, 3 , 3),
        (0,5, 5, 4 , 4),
        (0,5, 5, 5,  5),
        (0,5, 5, 6,  6),
        (0,5,30, 32-2-3-4-5, 32-2-3-4-5),
        (1, 30,10),
        (2,),
        (3,-1,0),
        (3,0,1),
        (3,1,2),
        (3,2,3),
        (3,3,4),
        (3,4,5),
        (3,5,6),
        )


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
                                                                                trial_epocs=TRIAL_EPOCS)

trainer.execute(periodSave=PERIOD_SAVE, periodNewSpace=PERIOD_NEWSPACE, periodSaveModel=PERIOD_SAVE_MODEL,
                epochs=EPOCHS, min_dt=MIN_DT, max_dt=MAX_DT, min_dt_2=MIN_DT_2, max_dt_2=MAX_DT_2)
