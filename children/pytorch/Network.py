import children.pytorch.Node as nd
import children.pytorch.Layer as ly
import children.pytorch.Functions as functions

import torch
import torch.nn as nn
import torch.tensor as tensor
import Factory.LayerFactory as factory
import Factory.TensorFactory as tensorFactory

import utilities.Graphs as Graphs
import torch.optim as optim

class Network(nn.Module):

    def __init__(self, adn,cudaFlag=True):
        super(Network, self).__init__()

        self.cudaFlag = cudaFlag
        self.adn = adn
        self.nodes = []
        #self.label = tensor([1], dtype=torch.long).cuda()
        self.label = tensorFactory.createTensor(body=[1], cuda=self.cudaFlag, requiresGrad=False)

        self.factory = factory.LayerGenerator(cuda=self.cudaFlag)
        self.foward_value = None

        self.__createStructure()
        self.total_value = 0
        self.history_loss = []
        self.__accumulated_loss = 0

    def setAttribute(self, name, value):
        setattr(self,name,value)

    def __getAttribute(self, name):

        attribute = None
        try:
            attribute = getattr(self, name)
        except AttributeError:
            pass

        return attribute

    def deleteAttribute(self, name):
        try:
            delattr(self, name)
        except AttributeError:
            pass

    def executeLayer(self, layerName, x):

        layer = self.__getAttribute(layerName)

        if layer is not None:

            return layer(x)

        else:

            return x

    def __createStructure(self):

        graph = Graphs.Graph(True)

        amount_of_nodes = len(self.adn) + 1

        for i in range(0, amount_of_nodes):
            self.nodes.append(nd.Node())

        for i in range(0, amount_of_nodes):

            if i < (amount_of_nodes - 1):
                graph.add_node(i, self.nodes[i])
                graph.add_node(i + 1, self.nodes[i + 1])
                graph.add_edges(i, [i+1])


        self.nodes = list(graph.key2node.values())
        #print("nodes: ", len(self.nodes))
        self.__assignLayers()

    def __assignLayers(self):

        self.nodes[0].objects.append(ly.Layer(node=self.nodes[0], value=None, propagate=functions.Nothing, cudaFlag=self.cudaFlag))

        for i in range(len(self.adn)):
            indexNode = i + 1
            tupleBody = self.adn[i]
            layer = self.factory.findValue(tupleBody)
            layer.node = self.nodes[indexNode]
            self.nodes[indexNode].objects.append(layer)
            attributeName = "layer"+str(indexNode)
            self.setAttribute(attributeName, layer.object)

    def Acumulate_der(self, n, peso=1):

        for i in range(len(self.nodes)-1):
            layer = self.nodes[i].objects[0]
            biasDer = layer.getBiasDer()
            filterDer = layer.getFilterDer()

            if biasDer is not None:
                layer.bias_der_total += (biasDer / n) * peso

            if filterDer is not None:
                layer.filter_der_total += (filterDer / n) * peso

        self.total_value += ((self.__getLossLayer().value)/n).item()


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

        labelTensor = tensorFactory.createTensor(body=[label.item()], cuda=self.cudaFlag, requiresGrad=False)
        self.assignLabels(labelTensor)

        self.nodes[0].objects[0].value = image.view(1, 3, image.shape[1], image.shape[2])


        self(self.nodes[0].objects[0].value)

        return self.getProbability()


    def getProbability(self):

        value = self.__getLayerProbability().value
        #print("output linear: ", value)
        sc = value[0][0]
        sn = value[0][1]

        p = torch.exp(sc)/(torch.exp(sc) + torch.exp(sn))

        return p.item()

    def Train(self, dataElement, peso, n):

        self.nodes[0].objects[0].value = dataElement
        self.updateGradFlag(True)
        self(dataElement)
        #self.__doFoward()
        self.__doBackward()
        self.updateGradFlag(False)
        #self.Acumulate_der(n, peso)
        #self.Reset_der()

    def Training(self, data, labels=None, dt=0.1, p=1):

            self.optimizer = optim.SGD(self.parameters(), lr=dt, momentum=0)
            dataGenerator = data

            i=0
            while i < p:

                
                if self.cudaFlag == True:
                    inputs, labels_data = dataGenerator.data[0].cuda(), dataGenerator.data[1].cuda()
                else:
                    inputs, labels_data = dataGenerator.data[0], dataGenerator.data[1]

                self.assignLabels(labels_data)
                self.total_value = 0
                self.optimizer.zero_grad()
                self.Train(inputs, 1, 1)
                self.optimizer.step()
                
                self.total_value = self.__getLossLayer().value.item()
                self.__accumulated_loss += self.total_value

                self.history_loss.append(self.total_value)

                dataGenerator.update()

                i=i+1

    def getAverageLoss_test(self, iterations):

        average = self.__accumulated_loss / iterations

        self.__accumulated_loss = 0

        return average
    
    def getAverageLoss(self, iterations):

        last_x = self.history_loss[-iterations:]

        accum = 0

        for loss in last_x:
            accum += loss
        
        return (accum/iterations)

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


    def forward(self, x):
        self.__doFoward()

    def __doFoward(self):

        functions.Propagation(self.__getLossLayer())

    def __doBackward(self):

        self.__getLossLayer().value.backward()


    def __getLossLayer(self):

        return self.nodes[len(self.nodes)-1].objects[0]

    def __getLayerProbability(self):

        return self.nodes[len(self.nodes)-2].objects[0]


    def getLossArray(self):

        value = self.history_loss.copy()
        self.history_loss = []
        return value

    def clone(self):

        newObjects = []
        newADN = tuple(list(self.adn))

        network = Network(newADN,cudaFlag=self.cudaFlag)

        for i in range(len(self.nodes) - 1):
            layerToClone = self.nodes[i].objects[0]
            layer = network.nodes[i].objects[0]

            if layerToClone.getBias() is not None:
                layer.setBias(layerToClone.getBias().clone())

            if layerToClone.getFilter() is not None:
                layer.setFilter(layerToClone.getFilter().clone())

        return network

    def __printGrad(self):
        if self.nodes[1].objects[0].getFilter().grad is not None:
            layer1 = self.nodes[1].objects[0]
            layer2 = self.nodes[2].objects[0]
            print("shape filter grad: ", layer1.getFilter().grad.shape)
            print("shape bias grad: ", layer1.getFilter().grad.shape)
            print("shape filter 2 grad: ", layer2.getFilter().grad.shape)
            print("shape bias 2 grad: ", layer2.getFilter().grad.shape)

    def generateEnergy(self, dataGen):

        accuracy = 0
        print("generate energy")
        with torch.no_grad():

            total = 0
            correct = 0

            for i, data in enumerate(dataGen._testloader):

                if self.cudaFlag == True:
                    inputs, labels = data[0].cuda(), data[1].cuda()
                else:
                    inputs, labels = data[0], data[1]
                
                self.assignLabels(labels)
                self.nodes[0].objects[0].value = inputs*255 # DESNORMALIZAR!
                self(self.nodes[0].objects[0].value)

                linearValue = self.__getLayerProbability().value

                _, predicted = torch.max(linearValue.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
        
        return accuracy