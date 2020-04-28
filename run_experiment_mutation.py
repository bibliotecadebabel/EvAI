from TestNetwork.commands import CommandCreateDataGen, CommandExperimentMutation
from DNA_conditions import max_layer
from DNA_creators import Creator
from DNA_Graph import DNA_Graph

###### EXPERIMENT SETTINGS ######

# Every PERDIOD_SAVE iterations, all DNA and its energies will be stored in the database.
PERIOD_SAVE = 10 

# Every PERDIOD_NEWSPACE iterations, a new DNA GRAPH will be generated with the dna of the lowest energy network as center.
PERIOD_NEWSPACE = 200 

# After TOTAL_ITERATIONS, the experiment will stop.
TOTAL_ITERATIONS = 10000

# dt parameter
DT = 0.01

# CUDA parameter (true/false)
CUDA = True

# MAX LAYER MUTATION (CONDITION)
MAX_LAYER = 3

# TEST_NAME, the name of the experiment (unique)
TEST_NAME = "test-3"


def DNA_test_i(x,y):
    def condition(DNA):
        output=True
        if DNA:
            for num_layer in range(0,len(DNA)-3):
                layer=DNA[num_layer]
                x_l=layer[3]
                y_l=layer[4]
                output=output and (x_l<x) and (y_l<y)
        if output:
            return max_layer(DNA,MAX_LAYER)
    center=((0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y), (1, 5, 2), (2,))
    version='inclusion'
    space=space=DNA_Graph(center,2,(x,y),condition,(0,(0,0,1,1),(0,1,0,0),(1,0,0,0)),version)
    return space


dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=CUDA)
dataCreator.execute(2, 200)
dataGen = dataCreator.returnParam()

space = DNA_test_i(dataGen.size[1], dataGen.size[2])


trainer = CommandExperimentMutation.CommandExperimentMutation(space=space, dataGen=dataGen, testName=TEST_NAME, cuda=CUDA)
                                                                
trainer.execute(periodSave=PERIOD_SAVE, periodNewSpace=PERIOD_NEWSPACE, totalIterations=TOTAL_ITERATIONS, dt=DT)