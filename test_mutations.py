from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
import children.pytorch.MutateNetwork as MutateNetwork
import children.pytorch.Network as nw

CUDA = False

def Test(dataGen, space):
    print("type=",space.type)
    global CUDA
    for node in space.objects:
        parentADN = space.node2key(node)

        parentNetwork = nw.Network(parentADN, cudaFlag=CUDA)

        print("Training PARENT Network | ADN = ", parentNetwork.adn)
        parentNetwork.Training(data=dataGen.data[0], labels=dataGen.data[1], dt=0.01, p=600)

        for nodeKid in node.kids:

            kidADN = space.node2key(nodeKid)

            kidNetwork = MutateNetwork.executeMutation(parentNetwork, kidADN)

            print("Training KID Network | ADN = ", kidNetwork.adn)
            kidNetwork.Training(data=dataGen.data[0], labels=dataGen.data[1], dt=0.01, p=600)


S=100
Comp=2
dataGen=GeneratorFromImage.GeneratorFromImage(Comp, S, cuda=CUDA)
dataGen.dataConv2d()

x = dataGen.size[1]
y = dataGen.size[2]
ks=[2]

center_filters= ((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
center_kernel = ((0, 3, int(2*ks[0]), 2, 2),(0, int(2*ks[0]),ks[0], x-1, y-1), (1, ks[0], 2), (2,))
space_filters=DNA_Graph(center_filters,5,(x,y))
space_kernel=DNA_Graph(center_kernel,5,(x,y),(0,(0,1,0,0)))

Test(dataGen, space_filters)