from TestNetwork.commands import CommandCreateDataGen, CommandExperimentCifar_Restarts
from DNA_conditions import max_layer,max_filter
from DNA_creators import Creator
from DNA_Graph import DNA_Graph
from DNA_creator_duplicate_clone import Creator_from_selection_clone as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector

###### EXPERIMENT SETTINGS ######

# BATCH SIZE
BATCH_SIZE = 25

# DATA SOURCE ('default' -> Pikachu, 'cifar' -> CIFAR)
DATA_SOURCE = 'cifar'

# Every PERDIOD_SAVE iterations, all DNA and its energies will be stored in the database.
PERIOD_SAVE = 1

# Every PERDIOD_NEWSPACE iterations, a new DNA GRAPH will be generated with the dna of the lowest energy network as center.
PERIOD_NEWSPACE = 1 

# Every PERIOD_SAVE_MODEL iterations, the best network (current center) will be stored on filesystem
PERIOD_SAVE_MODEL = 10

# After TOTAL_ITERATIONS, the experiment will stop.
TOTAL_ITERATIONS = 100

# dt parameter
DT = 0.001

# min_dt parameter
MIN_DT = 0.001

# CUDA parameter (true/false)
CUDA = True

# MAX LAYER MUTATION (CONDITION)
MAX_LAYERS = 15

# MAX FILTERS MUTATION (CONDITION)
MAX_FILTERS = 70

# TEST_NAME, the name of the experiment (unique)
TEST_NAME = "cifar_test_restart-clone_1"


def DNA_Creator_s(x,y):
    def condition(DNA):
        return max_filter(max_layer(DNA,MAX_LAYERS),MAX_FILTERS)

    center=((-1,1,3,x,y),(0,3, 15, 3 , 3),(0,18, 15, 3,  3),(0,33, 15, x, y),(1, 15, 10),
             (2,),(3,-1,0),(3,0,1),(3,-1,1),(3,1,2),(3,0,2),(3,-1,2),(3,2,3),(3,3,4))

    selector = None
    selector=random_selector()
    selector.update(center)
    actions=selector.get_predicted_actions()
    version='final'
    #actions = ((0, (0,1,0,0)), (1, (0,1,0,0)), (0, (1,0,0,0)))
    space=DNA_Graph(center,1,(x,y),condition,actions
        ,version,Creator_s)

    return [space, selector]

dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=CUDA)
dataCreator.execute(compression=2, batchSize=BATCH_SIZE, source=DATA_SOURCE)
dataGen = dataCreator.returnParam()

stop = False
space = None
selector = None
while stop == False:
    space, selector = DNA_Creator_s(dataGen.size[1], dataGen.size[2])

    for node in space.objects:

        adn = space.node2key(node)

        if str(adn) == str(space.center):

            if len(node.kids) > 0:
                stop = True
    


trainer = CommandExperimentCifar_Restarts.CommandExperimentCifar_Restarts(space=space, dataGen=dataGen, testName=TEST_NAME,selector=selector, cuda=CUDA)
                                                                
trainer.execute(periodSave=PERIOD_SAVE, periodNewSpace=PERIOD_NEWSPACE, totalIterations=TOTAL_ITERATIONS, base_dt=DT, min_dt=MIN_DT, periodSaveModel=PERIOD_SAVE_MODEL)
