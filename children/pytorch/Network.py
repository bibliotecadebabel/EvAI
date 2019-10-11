import children.pytorch.Node as nd
import children.pytorch.Layer as ly
import children.pytorch.Functions as functions

import torch
import torch.nn as nn
import torch.tensor as tensor

import utilities.Graphs as Graphs

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
        
        graph = Graphs.Graph()

        amount_of_nodes = 4

        for i in range(0, amount_of_nodes):
            self.nodes.append(nd.Node())

        for i in range(0, amount_of_nodes):

            if i < (amount_of_nodes - 1):
                graph.add_node(i, self.nodes[i])
                graph.add_node(i + 1, self.nodes[i + 1])
                graph.add_edges(i, [i+1])


        self.nodes = list(graph.key2node.values())
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

    def Acumulate_der(self, n, peso=1):

        for i in range(len(self.nodes)-1):
            layer = self.nodes[i].objects[0]
            biasDer = layer.getBiasDer()
            filterDer = layer.getFilterDer()
                
            if biasDer is not None:
                layer.bias_der_total += (biasDer / n) * peso

            if filterDer is not None:
                layer.filter_der_total += (filterDer / n) * peso
    
    def Regularize_der(self):

        for node in self.nodes[:-1]:
            layer = node.objects[0]

            bias = layer.getBias()
            filters = layer.getFilter()

            if bias is not None and layer.bias_der_total is not None:
                layer.bias_der_total = layer.bias_der_total + bias

            if filters is not None and layer.filter_der_total is not None:
                layer.filter_der_total = layer.filter_der_total + filters

    def Reset_der(self):

        for node in self.nodes:
            layer = node.objects[0]

            if layer.object is not None:
                layer.object.zero_grad()

    def Reset_der_total(self):

        for node in self.nodes:
            layer = node.objects[0]

            if layer.bias_der_total is not None:
                layer.bias_der_total = layer.bias_der_total * 0

            if layer.filter_der_total is not None:
                layer.filter_der_total = layer.filter_der_total * 0
    

    
    def assignLabels(self, label):

        for node in self.nodes:
            node.objects[0].label = label

    def Update(self, dt):

        
        for node in self.nodes[:-1]:
            layer = node.objects[0]

            if layer.getFilter() is not None:
                layer.getFilter().data -= (layer.filter_der_total * dt)

            if layer.getBias() is not None:
                layer.getBias().data -= (layer.bias_der_total * dt)
        
    def Predict(self, image):
        #self.assignLabels("c")
        self.nodes[0].objects[0].value = image[0]

        functions.Propagation(self.nodes[3].objects[0])

        return self.nodes[3].objects[0].value


    def Train(self, dataElement, peso, n):
        self.nodes[0].objects[0].value = dataElement[0]
        #self.nodes[3].objects[0].label = dataElement[1]

        self.assignLabels(dataElement[1])

        functions.Propagation(self.nodes[3].objects[0])
        self.nodes[3].objects[0].value.backward()
        print("value nodo 3: ", self.nodes[3].objects[0].value)
        self.Acumulate_der(n, peso)


    def Training(self, data, dt=0.1, p=0.99):
            n = len(data) * 5/4
            peso = len(data) / 4

            #self.Train(data[0], peso, n)

            i=0
            while i < p:
                if i % 10==0:
                    #print(i)
                    print(self.nodes[3].objects[0].value)
                self.Reset_der_total()
                self.Train(data[0], peso, n)

                for image in data[1:]:
                    self.Train(image, 1, n)

                self.Regularize_der()
                self.Update(dt)
                i=i+1