import children.pytorch.Node as nd
import children.pytorch.Layer as ly
import children.pytorch.Functions as functions
import children.pytorch.NetworkAbastract as na
import const.propagate_mode as const
import const.datagenerator_type as datagen_type

import torch
import torch.nn as nn
import torch.tensor as tensor

import utilities.Graphs as Graphs
import torch.optim as optim

class Network(nn.Module, na.NetworkAbstract):

    def __init__(self, adn, cudaFlag=True, momentum=0.0):
        nn.Module.__init__(self)
        na.NetworkAbstract.__init__(self,adn=adn, cuda=cudaFlag, momentum=momentum)

        self.__lenghNodes = 0
        self.__conv2d_propagate_mode = const.CONV2D_DEFAULT
        self.__accumulated_loss = 0
        self.__accuracy = 0
        self.createStructure()

    def createStructure(self):

        graph = Graphs.Graph(True)

        self.__generateLengthNodes()

        for i in range(0, self.__lenghNodes):
            self.nodes.append(nd.Node())

        for i in range(0, self.__lenghNodes):

            if i < (self.__lenghNodes - 1):
                graph.add_node(i, self.nodes[i])
                graph.add_node(i + 1, self.nodes[i + 1])
                graph.add_edges(i, [i+1])


        self.nodes = list(graph.key2node.values())

        self.__assignLayers()

    def __assignLayers(self):

        self.nodes[0].objects.append(ly.Layer(node=self.nodes[0], value=None, propagate=functions.Nothing, cudaFlag=self.cudaFlag))

        indexNode = 1
        for adn in self.adn:
            tupleBody = adn

            if tupleBody[0] >= 0 and tupleBody[0] <= 2:
                layer = self.factory.findValue(tupleBody, propagate_mode=self.__conv2d_propagate_mode)
                layer.node = self.nodes[indexNode]
                self.nodes[indexNode].objects.append(layer)
                attributeName = "layer"+str(indexNode)
                self.setAttribute(attributeName, layer.object)

                indexNode += 1

            elif tupleBody[0] == 3:
                input_node = tupleBody[1]+1
                target_node = tupleBody[2]+1
                self.nodes[target_node].objects[0].other_inputs.append(self.nodes[input_node].objects[0])

    def __generateLengthNodes(self):

        for i in range(len(self.adn)):
            
            tupleBody = self.adn[i]

            if tupleBody[0] >= 0 and tupleBody[0] <= 2:
                self.__lenghNodes += 1
            
            if tupleBody[0] == 3:
                self.__conv2d_propagate_mode = const.CONV2D_MULTIPLE_INPUTS
        
        self.__lenghNodes += 1

    def Predict(self, image, label):

        labelTensor = na.tensorFactory.createTensor(body=[label.item()], cuda=self.cudaFlag, requiresGrad=False)
        self.assignLabels(labelTensor)

        value = image.view(1, 3, image.shape[1], image.shape[2])

        if self.cudaFlag == True:
            value = value.cuda()
        
        self.nodes[0].objects[0].value = value
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
        self.__doBackward()
        self.updateGradFlag(False)


    def Training(self, data, labels=None, dt=0.1, p=1, full_database=False):

            self.optimizer = optim.SGD(self.parameters(), lr=dt, momentum=self.momentum)
            
            if full_database == False:
                self.__defaultTraining(dataGenerator=data, p=p)
            else:
                self.__fullDatabaseTraining(dataGenerator=data, p=p)


    def __defaultTraining(self, dataGenerator, p):
            
            i=0
            while i < p:
                #if i == 0:
                #    print("Accuracy before first iteration=", self.generateEnergy(data))
    
                #if i % 100 == 99:
                #    print("Loss=", self.getAverageLoss(50), "i=", i)

                #if i == 1:
                #    print("Loss=", self.total_value, "- i=", i)
                #if i % 100 == 99:
                #    print("Loss=", self.total_value, "- i=", i)

                if self.cudaFlag == True:
                    inputs, labels_data = dataGenerator.data[0].cuda(), dataGenerator.data[1].cuda()
                else:
                    inputs, labels_data = dataGenerator.data[0], dataGenerator.data[1]
                
                self.__doTraining(inputs=inputs, labels_data=labels_data)

                dataGenerator.update()

                i=i+1
    
    def __fullDatabaseTraining(self, dataGenerator, p):
        epoch=0
        
        if dataGenerator.type == datagen_type.DATABASE_IMAGES:
            
            while epoch < p:
                
                for i, data in enumerate(dataGenerator._trainoader):
                    
                    if self.cudaFlag == True:
                        inputs, labels_data = data[0].cuda(), data[1].cuda()
                    else:
                        inputs, labels_data = data[0], data[1]

                    inputs = inputs * 255

                    self.__doTraining(inputs=inputs, labels_data=labels_data)
                
                epoch += 1

        elif dataGenerator.type == datagen_type.OWN_IMAGE:

            dataGenerator.generateDataBase()

            while epoch < p:
                
                i = 0
                for data in dataGenerator.batch(dataGenerator.batch_size):
                    
                    if self.cudaFlag == True:
                        inputs, labels_data = data[0].cuda(), data[1].cuda()
                    else:
                        inputs, labels_data = data[0], data[1]

                    self.__doTraining(inputs=inputs, labels_data=labels_data)

                    #if i % 10 == 0:
                    #    print("[",epoch,", ", i," ,", self.total_value,"]")

                    i += 1

                epoch += 1
        
        else:
            print("ERROR UNKNOWN DATAGENERATOR TPYE")
            
    

    def __doTraining(self, inputs, labels_data):

        self.assignLabels(labels_data)
        self.total_value = 0
        self.optimizer.zero_grad()
        self.Train(inputs, 1, 1)
        self.optimizer.step()
        
        self.total_value = self.__getLossLayer().value.item()
        self.__accumulated_loss += self.total_value

        self.history_loss.append(self.total_value)

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
        
        self.__accuracy = accuracy
    
    def getAcurracy(self):

        return self.__accuracy