import Node as nd
import Layer as ly
import Functions as functions

import torch
import torch.nn as nn
import torch.tensor as tensor


class Network(nn.Module):

    def __init__(self, objects):
        super(Network, self).__init__()
        # objects [x, y, k]
        # x -> dimension x
        # y -> dimension y
        # k -> cantidad filtros
        self.objects = objects
        self.nodes = []
        self.label = tensor([0,1], dtype=torch.float32)

        self.foward_value = None

        self.__createStructure()
        self.total_value = 0


    def __createStructure(self):
        nodes = []

        nodes.append(nd.Node())
        nodes.append(nd.Node())
        nodes.append(nd.Node())
        nodes.append(nd.Node())
        #nodes.append(nd.Node())


        nodes[0].kids.append(nodes[1])
        nodes[1].kids.append(nodes[2])
        nodes[2].kids.append(nodes[3])
        #nodes[3].kids.append(nodes[4])

        nodes[1].parents.append(nodes[0])
        nodes[2].parents.append(nodes[1])
        nodes[3].parents.append(nodes[2])
        #nodes[4].parents.append(nodes[3])

        self.nodes = nodes
        self.__assignLayers()

    def __assignLayers(self):

        objectConv2d = nn.Conv2d(3, self.objects[2], self.objects[0], self.objects[1])
        objectLinear = nn.Linear(self.objects[2], 2)
        objectMSELoss = nn.MSELoss()

        valueLayerA = torch.rand(1, 3, self.objects[0], self.objects[1], dtype=torch.float32)

        self.nodes[0].objects.append(ly.Layer(node=self.nodes[0], value=valueLayerA, propagate=functions.Nothing))
        self.nodes[1].objects.append(ly.Layer(node=self.nodes[1], objectTorch=objectConv2d, propagate=functions.conv2d_propagate))
        self.nodes[2].objects.append(ly.Layer(node=self.nodes[2], objectTorch=objectLinear, propagate=functions.linear_propagate))
        self.nodes[3].objects.append(ly.Layer(node=self.nodes[3], objectTorch=objectMSELoss, label=self.label, propagate=functions.MSEloss_propagate))


    def print_params(self):
        
        i = 1
        for node in self.nodes:
            if len(node.objects) > 0 and node.objects[0].object is not None and len(list(node.objects[0].object.parameters())) > 0:
                print("PARAMETROS LAYER ",i)
                for param in node.objects[0].object.parameters():
                    print(type(param.data), param.size())
                    print("grad: ", param.grad)
                i += 1