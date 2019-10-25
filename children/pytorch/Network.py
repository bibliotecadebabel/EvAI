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
        self.label = tensor([1], dtype=torch.long).cuda()

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

        objectConv2d = nn.Conv2d(3, self.objects[2], self.objects[0], self.objects[1]).cuda()
        objectLinear = nn.Linear(self.objects[2], 2).cuda()
        objectMSELoss = nn.CrossEntropyLoss().cuda()

        valueLayerA = torch.rand(1, 3, self.objects[0], self.objects[1], dtype=torch.float32).cuda()
        valueLayerA = valueLayerA
        valueLayerConv2d = torch.rand(1, self.objects[2], 1, 1, dtype=torch.float32, requires_grad=True).cuda()
        valueLayerLinear = torch.rand(2, dtype=torch.float32, requires_grad=True).cuda()

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

        self.total_value += ((self.nodes[3].objects[0].value)/n).item()
        #self.total_value += self.nodes[3].objects[0].value.item()

    def Regularize_der(self):

        for node in self.nodes[:-1]:
            layer = node.objects[0]

            bias = layer.getBias()
            filters = layer.getFilter()

            if bias is not None and layer.bias_der_total is not None:
                layer.bias_der_total = layer.bias_der_total + (bias/1000)

            if filters is not None and layer.filter_der_total is not None:
                layer.filter_der_total = layer.filter_der_total + (filters/1000)

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
                    #print("grad: ", layer.value.grad)
                    layer.value.grad.data.zero_()

             

    def Reset_der_total(self):

        for node in self.nodes:
            layer = node.objects[0]

            if layer.bias_der_total is not None:
                layer.bias_der_total = layer.bias_der_total * 0

            if layer.filter_der_total is not None:
                layer.filter_der_total = layer.filter_der_total * 0

        self.total_value  = 0
    

    
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
        
    def Predict(self, image, label):
        #self.assignLabels(torch.tensor([0], dtype=torch.long))
        
        labelTensor = torch.tensor([label.item()]).long().cuda()
        self.assignLabels(labelTensor)
        self.nodes[0].objects[0].value = image.view(1, 3, self.objects[0], self.objects[1])

        functions.Propagation(self.nodes[3].objects[0])

        #print("prob of filters#", self.objects[2],": ", self.getProbability())
        return self.getProbability()


    def getProbability(self):

        value = self.nodes[2].objects[0].value
        #print("output linear: ", value)
        sc = value[0][0]
        sn = value[0][1]

        p = torch.exp(sc)/(torch.exp(sc) + torch.exp(sn))

        return p.item()

    def Train(self, dataElement, peso, n):

        self.nodes[0].objects[0].value = dataElement
        self.updateGradFlag(True)
        functions.Propagation(self.nodes[3].objects[0])
        #self.showParameters()
        self.nodes[3].objects[0].value.backward()
        self.updateGradFlag(False)

        self.Acumulate_der(n, peso)
        self.Reset_der()

    def Training(self, data, labels, dt=0.1, p=2):
            n = len(data) * 5/4
            peso = len(data) / 4

            i=0
            while i < p:
                if i % 2000 == 1999:
                    print("i=", i+1)
                    print("Energy of #",self.objects[2],": ", self.total_value )
                    #self.total_value = 0
                    #print("prob: ", self.Predict(data[0], labels[0]))
                    #print("prob of filters #", self.objects[2], " :", self.getProbability())
                    pass
                
                self.assignLabels(labels)
                self.Reset_der_total()
                self.Train(data, 1, 1)

                #for image in data[1:]:
                    #self.Train(image, 1, n)

                #self.Regularize_der()
                self.Update(dt)
                #self.Predict(data[0])
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

    def __removeGrad(self):

        pass

    def addFilters(self):
        layerConv2d = self.nodes[1].objects[0]
        layerLinear = self.nodes[2].objects[0]

        layerConv2d.getFilter().grad = None
        layerConv2d.getBias().grad = None
        layerLinear.getFilter().grad = None
        layerLinear.getBias().grad = None

        shapeFilterConv2d = layerConv2d.getFilter().shape
        shapeFilterLinear = layerLinear.getFilter().shape
        shapeValueConv2d = layerConv2d.value.shape

        self.updateGradFlag(False)
        layerConv2d.value.requires_grad = False 

        layerConv2d.getFilter().resize_(shapeFilterConv2d[0]+1, 3, shapeFilterConv2d[2], shapeFilterConv2d[3])
        layerConv2d.getBias().resize_(shapeFilterConv2d[0]+1)
        
        layerConv2d.value.resize_(shapeValueConv2d[0], shapeValueConv2d[1]+1, shapeValueConv2d[2], shapeValueConv2d[3])
        

        layerLinear.getFilter().resize_(2, shapeFilterLinear[1]+1)
 
        layerLinear.filter_der_total = 0
        layerLinear.bias_der_total = 0
        layerConv2d.filter_der_total = 0
        layerConv2d.bias_der_total = 0

        layerConv2d.getFilter()[shapeFilterConv2d[0]] = layerConv2d.getFilter()[shapeFilterConv2d[0]-1].clone()
        
        layerConv2d.getBias()[shapeFilterConv2d[0]] = layerConv2d.getBias()[shapeFilterConv2d[0]-1].clone()

        layerConv2d.value[shapeValueConv2d[0]-1][shapeValueConv2d[1]] = layerConv2d.value[shapeValueConv2d[0]-1][shapeValueConv2d[1]-1].clone()

        for i in range(layerLinear.getFilter().shape[0]):
            layerLinear.getFilter()[i][layerLinear.getFilter().shape[1]-1] = layerLinear.getFilter()[i][layerLinear.getFilter().shape[1]-2].clone()

        self.updateGradFlag(True)
        layerConv2d.value.requires_grad = True

    def removeFilter(self):
        
        layerConv2d = self.nodes[1].objects[0]
        layerLinear = self.nodes[2].objects[0]

        layerConv2d.getFilter().grad = None
        layerConv2d.getBias().grad = None
        layerLinear.getFilter().grad = None
        layerLinear.getBias().grad = None

        shapeFilterConv2d = layerConv2d.getFilter().shape
        shapeFilterLinear = layerLinear.getFilter().shape
        shapeValueConv2d = layerConv2d.value.shape

        self.updateGradFlag(False)
            
        layerConv2d.getFilter().resize_(shapeFilterConv2d[0]-1, 3, shapeFilterConv2d[2], shapeFilterConv2d[3])
        layerConv2d.getBias().resize_(shapeFilterConv2d[0]-1)
        
        layerConv2d.value.requires_grad = False
        layerConv2d.value.resize_(shapeValueConv2d[0], shapeValueConv2d[1]-1, shapeValueConv2d[2], shapeValueConv2d[3])
        layerConv2d.value.requires_grad = True

        layerLinear.getFilter().resize_(2, shapeFilterLinear[1]-1)
 
        layerLinear.filter_der_total = 0
        layerLinear.bias_der_total = 0
        layerConv2d.filter_der_total = 0
        layerConv2d.bias_der_total = 0

        self.updateGradFlag(True)
       





