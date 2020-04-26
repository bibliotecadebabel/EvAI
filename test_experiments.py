from TestNetwork.commands import CommandCreateDataGen, CommandExperimentMutation
from DNA_conditions import max_layer
from DNA_creators import Creator
from DNA_Graph import DNA_Graph


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
            return max_layer(DNA,2)
    center=((0, 3, 5, 3, 3),(0,8,5, x, y), (1, 5, 2), (2,))
    version='inclusion'
    space=space=DNA_Graph(center,2,(x,y),condition,(0,(0,0,1,1),(0,1,0,0),(1,0,0,0)),version)
    return space


dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=True)
dataCreator.execute(2, 200)
dataGen = dataCreator.returnParam()

space = DNA_test_i(dataGen.size[1], dataGen.size[2])

TEST_NAME = "real-test-1"
TEST_ID = 1

trainer = CommandExperimentMutation.CommandExperimentMutation(space=space, dataGen=dataGen, testName=TEST_NAME,
                                                                testId=TEST_ID, cuda=True)
                                                                
trainer.execute(periodSave=10, periodNewSpace=200, totalIterations=1000, dt=0.01)