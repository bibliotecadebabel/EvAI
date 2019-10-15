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
        
        graph = Graphs.Graph(True)

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
        valueLayerA = valueLayerA
        valueLayerConv2d = torch.rand(1, self.objects[2], 1, 1, dtype=torch.float32)
        valueLayerLinear = torch.rand(2, dtype=torch.float32)

        self.nodes[0].objects.append(ly.Layer(node=self.nodes[0], value=valueLayerA, propagate=functions.Nothing))
        self.nodes[1].objects.append(ly.Layer(node=self.nodes[1], objectTorch=objectConv2d, propagate=functions.conv2d_propagate, value=valueLayerConv2d))
        self.nodes[2].objects.append(ly.Layer(node=self.nodes[2], objectTorch=objectLinear, propagate=functions.linear_propagate, value=valueLayerLinear))
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

        for node in self.nodes[:-1]:
            layer = node.objects[0]

            #if layer.object is not None:
            #    layer.object.zero_grad()

            if layer.getBiasDer() is not None:
                layer.getBiasDer().data.zero_()
            
            if layer.getFilterDer() is not None:
                layer.getFilterDer().data.zero_()
            
        
        for node in self.nodes:
            layer = node.objects[0]

            if layer.value is not None:
                if layer.value.grad is not None:
                    layer.value.grad.data.zero_()

             

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
        self.assignLabels(torch.tensor([1,0], dtype=torch.float32))
        self.nodes[0].objects[0].value = image[0]

        functions.Propagation(self.nodes[3].objects[0])

        return self.getProbability()


    def getProbability(self):

        p = self.nodes[2].objects[0].value[0].item()

        if p < 0:
            p = 0
        elif p > 1:
            p = 1

        return p

    def Train(self, dataElement, peso, n):
        self.nodes[0].objects[0].value = dataElement[0]
        #self.nodes[3].objects[0].label = dataElement[1]

        self.assignLabels(dataElement[1])

        self.updateGradFlag(True)
        functions.Propagation(self.nodes[3].objects[0])
        self.nodes[3].objects[0].value.backward()
        self.updateGradFlag(False)

        self.Acumulate_der(n, peso)
        self.Reset_der()

    def Training(self, data, dt=0.1, p=0.99):
            n = len(data) * 5/4
            peso = len(data) / 4

            i=0
            while i < p:
                if i % 10==0:
                    print(i)
                    print("prob: ", self.getProbability())
                self.Train(data[0], peso, n)

                for image in data[1:]:
                    self.Train(image, 1, n)

                #self.Regularize_der()
                self.Update(dt)
                self.Reset_der_total()
                self.Predict(data[0])
                i=i+1
    
    def updateGradFlag(self, flag):

        for node in self.nodes[:-1]:

            layer = node.objects[0]

            if layer.getBias() is not None:
                layer.getBias().requires_grad = flag
            
            if layer.getBiasDer() is not None:
                layer.getBiasDer().requires_grad = flag
            
            if layer.getFilter() is not None:
                layer.getFilter().requires_grad = flag
            
            if layer.getFilterDer() is not None:
                layer.getFilterDer().requires_grad = flag
        
        '''
        for node in self.nodes:
            layer = node.objects[0]

            if layer.value is not None:
                layer.value.requires_grad = flag

        '''