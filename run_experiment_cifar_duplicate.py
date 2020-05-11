from TestNetwork.commands import CommandCreateDataGen, CommandExperimentCifar_duplicate
from DNA_conditions import max_layer
from DNA_Graph import DNA_Graph
from DNA_creator_duplicate import Creator_from_selection_duplicate as Creator_s
from utilities.Abstract_classes.classes.random_selector import random_selector

###### EXPERIMENT SETTINGS ######

# BATCH SIZE
BATCH_SIZE = 100

# DATA SOURCE ('default' -> Pikachu, 'cifar' -> CIFAR)
DATA_SOURCE = 'cifar'

# Every PERDIOD_SAVE iterations, all DNA and its energies will be stored in the database.
PERIOD_SAVE = 1

# Every PERDIOD_NEWSPACE iterations, a new DNA GRAPH will be generated with the dna of the lowest energy network as center.
PERIOD_NEWSPACE = 1 

# After TOTAL_ITERATIONS, the experiment will stop.
TOTAL_ITERATIONS = 20

# dt parameter
DT = 0.01

# CUDA parameter (true/false)
CUDA = True

# MAX LAYER MUTATION (CONDITION)
MAX_LAYER = 25

# TEST_NAME, the name of the experiment (unique)
TEST_NAME = "test-cifar-fulldb_duplicate"


def DNA_Creator_s(x,y):
    def condition(DNA):
        output=True
        if DNA:
            for num_layer in range(0,len(DNA)-3):
                layer=DNA[num_layer]
                
                if layer[0] == 0:
                    x_l=layer[3]
                    y_l=layer[4]
                    output=output and (x_l<x) and (y_l<y)
        if output:
            return max_layer(DNA,MAX_LAYER)
    center=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0,5, 5, x-2, y-2),
            (1, 5, 10), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3))
    selector=random_selector()
    selector.update(center)
    actions=selector.get_predicted_actions()
    version='final'
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
    


trainer = CommandExperimentCifar_duplicate.CommandExperimentCifar_Duplicate(space=space, dataGen=dataGen, testName=TEST_NAME,selector=selector, cuda=CUDA)
                                                                
trainer.execute(periodSave=PERIOD_SAVE, periodNewSpace=PERIOD_NEWSPACE, totalIterations=TOTAL_ITERATIONS, dt=DT)
