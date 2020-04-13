from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
from DNA_conditions import max_layer
from DNA_creators import Creator

import children.pytorch.MutateNetwork as MutateNetwork
import children.pytorch.Network as nw

CUDA = False


def generateSpace(x,y):
    def condition(DNA):
        output=True
        if DNA:
            for num_layer in range(0,len(DNA)-3):
                layer=DNA[num_layer]
                x_l=layer[3]
                y_l=layer[4]
                output=output and (x_l<x) and (y_l<y)
        if output:
            return max_layer(DNA,10)
    center=((0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y), (1, 5, 2), (2,))
    version='inclusion'
    space=space=DNA_Graph(center,4,(x,y),condition,(0,(0,0,1,1),(0,1,0,0),(1,0,0,0)),version)
    return space

def Test(dataGen, space):

    global CUDA
    i = 1
    for node in space.objects:
        parentADN = space.node2key(node)

        parentNetwork = nw.Network(parentADN, cudaFlag=CUDA)

        print("START TRAINING PARENT AND KIDS #", i)
        print("Training PARENT Network | ADN = ", parentNetwork.adn)

        for iteration in range(1, 201):
            parentNetwork.Training(data=dataGen.data[0], labels=dataGen.data[1], dt=0.01, p=1)
            
            if iteration % 100 == 0 or iteration == 1:
                print("L=", parentNetwork.total_value," - i=", iteration)

        for nodeKid in node.kids:

            kidADN = space.node2key(nodeKid)

            kidNetwork = MutateNetwork.executeMutation(parentNetwork, kidADN)

            print("Training KID Network | ADN = ", kidNetwork.adn)

            for iteration in range(1, 201):
                kidNetwork.Training(data=dataGen.data[0], labels=dataGen.data[1], dt=0.01, p=1)

                if iteration % 100 == 0 or iteration == 1:
                    print("L=", kidNetwork.total_value," - i=", iteration)
        
        print("FINISH TRAINING PARENT AND KIDS #",i,"\n \n")
        
        i += 1


S=100
Comp=2
dataGen=GeneratorFromImage.GeneratorFromImage(Comp, S, cuda=CUDA)
dataGen.dataConv2d()

x = dataGen.size[1]
y = dataGen.size[2]

space = generateSpace(x, y)

Test(dataGen, space)