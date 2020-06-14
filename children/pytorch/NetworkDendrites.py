import children.pytorch.Node as nd
import children.pytorch.Layer as ly
import children.pytorch.Functions as functions
import children.pytorch.NetworkAbastract as na
import const.propagate_mode as const
import const.datagenerator_type as datagen_type
import const.versions as directions_version
import utilities.Augmentation as Augmentation

import torch
import torch.nn as nn
import torch.tensor as tensor

import utilities.Graphs as Graphs
import torch.optim as optim

import os
import time

class Network(nn.Module, na.NetworkAbstract):

    def __init__(self, adn, cudaFlag=True, momentum=0.0, weight_decay=0.0, 
                enable_activation=True, enable_track_stats=True, dropout_value=0, 
                dropout_function=None, enable_last_activation=True, version=None, eps_batchnorm=None):
        nn.Module.__init__(self)
        na.NetworkAbstract.__init__(self,adn=adn, cuda=cudaFlag, momentum=momentum, weight_decay=weight_decay, 
                                enable_activaiton=enable_activation, enable_track_stats=enable_track_stats, dropout_value=dropout_value,
                                enable_last_activation=enable_last_activation)
        
        self.dropout_function = dropout_function
        self.version = version
        self.eps_batchnorm = eps_batchnorm

        if dropout_function == None:
            self.dropout_function = self.__defaultDropoutFunction
        
    
        self.__lenghNodes = 0
        self.__total_layers = 0
        self.__accumulated_loss = 0
        self.__accuracy = 0

        self.__conv2d_propagate_mode = const.CONV2D_MULTIPLE_INPUTS

        if self.version == directions_version.H_VERSION or self.version == directions_version.CONEX_VERSION:
            self.__conv2d_propagate_mode = const.CONV2D_PADDING

        self.createStructure()
        self.__currentEpoch = 0
        self.optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=self.momentum, weight_decay=self.weight_decay)  
          
    def __defaultDropoutFunction(self, base_p, total_layers, index_layer, isPool=False):

        return base_p

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
        index_layer = 0

        for adn in self.adn:

            tupleBody = adn

            if tupleBody[0] != -1 and tupleBody[0] != 3:
                
                layer = self.factory.findValue(tupleBody, propagate_mode=self.__conv2d_propagate_mode, 
                                        enable_activation=self.enable_activation)

                layer.node = self.nodes[indexNode]
                self.nodes[indexNode].objects.append(layer)
                attributeName = "layer"+str(indexNode)
                self.setAttribute(attributeName, layer.object)
                
                if index_layer == self.__total_layers - 2 and self.enable_last_activation == False:
                    layer.enable_activation = False

                if tupleBody[0] == 0 or tupleBody[0] == 1:
                    
                    if len(tupleBody) > 5:
                        dropout_value = self.dropout_function(self.dropout_value, self.__total_layers, index_layer, True)
                    else:
                        dropout_value = self.dropout_function(self.dropout_value, self.__total_layers, index_layer, False)

                    layer.dropout_value = dropout_value
                    conv2d_dropout = torch.nn.Dropout2d(p=layer.dropout_value)

                    if self.cudaFlag == True:
                        conv2d_dropout = conv2d_dropout.cuda()

                    layer.setDropoutObject(conv2d_dropout)
                    attributeName_dropout = "layer_dropout"+str(indexNode)
                    self.setAttribute(attributeName_dropout, layer.getDropoutObject())
                    index_layer += 1

                if tupleBody[0] == 0:

                    if self.eps_batchnorm == None:
                        conv2d_batchnorm = torch.nn.BatchNorm2d(tupleBody[2], track_running_stats=self.enable_track_stats)
                    else:
                        conv2d_batchnorm = torch.nn.BatchNorm2d(tupleBody[2], track_running_stats=self.enable_track_stats, eps=self.eps_batchnorm)

                    if self.cudaFlag == True:
                        conv2d_batchnorm = conv2d_batchnorm.cuda()
                    
                    layer.setBatchNormObject(conv2d_batchnorm)
                    attributeName_batch = "layer_batchnorm"+str(indexNode)
                    self.setAttribute(attributeName_batch, layer.getBatchNormObject())

                    if len(tupleBody) > 5:
                        conv2d_pool = torch.nn.MaxPool2d((tupleBody[5], tupleBody[5]), stride=None, ceil_mode=True)

                        if self.cudaFlag == True:
                            conv2d_pool = conv2d_pool.cuda()

                        layer.setPool(conv2d_pool)
                        attributeName_pool = "layer_pool"+str(indexNode)
                        self.setAttribute(attributeName_pool, layer.getPool())

                indexNode += 1

            elif tupleBody[0] == 3:
                input_node = tupleBody[1]+1
                target_node = tupleBody[2]+1
                self.nodes[target_node].objects[0].other_inputs.append(self.nodes[input_node].objects[0])

    def __generateLengthNodes(self):

        self.__total_layers = 0
        for i in range(len(self.adn)):

            tupleBody = self.adn[i]

            if tupleBody[0] != -1 and tupleBody[0] != 3:
                self.__lenghNodes += 1

                if tupleBody[0] == 0 or tupleBody[0] == 1:
                    self.__total_layers += 1

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

    def __doTraining(self, inputs, labels_data):

        self.__getLossLayer().setRicap(None)
        self.assignLabels(labels_data)
        self.total_value = 0
        self.optimizer.zero_grad()
        self.Train(inputs, 1, 1)
        self.optimizer.step()

        self.total_value = self.__getLossLayer().value.item()
        self.__accumulated_loss += self.total_value

        self.history_loss.append(self.total_value)
    
    def __doTrainingRICAP(self, inputs, labels_data, ricap : Augmentation.Ricap):

        self.__getLossLayer().setEnableRicap(True)
        self.assignLabels(labels_data)
        self.total_value = 0
        self.optimizer.zero_grad()

        patched_images = ricap.doRicap(inputs=inputs, target=labels_data, cuda=self.cudaFlag)        
        self.__getLossLayer().setRicap(ricap)
        self.Train(patched_images, 1, 1)
        self.optimizer.step()

        self.total_value = self.__getLossLayer().value.item()
        self.__accumulated_loss += self.total_value

        self.history_loss.append(self.total_value)

    def TrainingALaising(self, dataGenerator, epochs, alaising_object, period_show_accuracy):
        epoch = 0

        steps_per_epoch = len(dataGenerator._trainoader)
        print("steps_per_epoch= ", steps_per_epoch)
        print("weight decay= ", self.weight_decay)
        print("momentum= ", self.momentum)
        period_print = steps_per_epoch // 4

        while epoch < epochs:

            dt_array = alaising_object.get_increments(size=steps_per_epoch)

            if epoch % period_show_accuracy == 0:

                self.generateEnergy(dataGen=dataGenerator)
                print("ACCURACY= ", self.getAcurracy())

            print("current_max: ", alaising_object.current_max," - current_min: ", alaising_object.current_min)
            for i, data in enumerate(dataGenerator._trainoader):

                self.optimizer = optim.SGD(self.parameters(), lr=dt_array[i],
                        momentum=self.momentum, weight_decay=self.weight_decay)

                if self.cudaFlag == True:
                    inputs, labels_data = data[0].cuda(), data[1].cuda()
                else:
                    inputs, labels_data = data[0], data[1]

                self.assignLabels(labels_data)

                self.total_value = 0
                self.optimizer.zero_grad()
                self.Train(inputs, 1, 1)
                self.optimizer.step()

                self.total_value = self.__getLossLayer().value.item()
                self.__accumulated_loss += self.total_value
                self.history_loss.append(self.total_value)

                if i % period_print == period_print - 1:
                    self.__printValues(epoch + 1, i, avg=period_print)

            epoch += 1

    def TrainingCosineLR(self, dataGenerator, dt_array, iteration):

        start_step = iteration
        print("start step=", start_step)

        total_steps = len(dataGenerator._trainoader)

        for _, data in enumerate(dataGenerator._trainoader):

            self.optimizer = optim.SGD(self.parameters(), lr=dt_array[start_step], momentum=self.momentum, weight_decay=self.weight_decay)

            if self.cudaFlag == True:
                inputs, labels_data = data[0].cuda(), data[1].cuda()
            else:
                inputs, labels_data = data[0], data[1]

            self.assignLabels(labels_data)
            self.total_value = 0
            self.optimizer.zero_grad()
            self.Train(inputs, 1, 1)
            self.optimizer.step()

            self.total_value = self.__getLossLayer().value.item()
            self.__accumulated_loss += self.total_value
            self.history_loss.append(self.total_value)

            start_step += 1

        print("end step=", start_step)
        print("end energy=", self.getAverageLoss(total_steps//4))

    def TrainingCosineLR_Restarts(self, dataGenerator, max_dt, min_dt, epochs, restart_dt=1, show_accuarcy=False, fileManager=None):

        print("momentum=", self.momentum)
        print("weight decay=", self.weight_decay)
        self.optimizer = optim.SGD(self.parameters(), lr=max_dt, momentum=self.momentum, weight_decay=self.weight_decay)
        total_steps = len(dataGenerator._trainoader)

        print_every = total_steps // 8
        epoch = 0

        #scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, total_steps * restart_dt, eta_min=min_dt)
        while epoch < epochs:
            start_time = time.time()
            #if epoch % restart_dt == 0:
            #    scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, total_steps * restart_dt, eta_min=min_dt)

            if show_accuarcy == True and epoch > 0:

                if epoch % 5 == 0:
                    self.generateEnergy(dataGen=dataGenerator)
                    print("ACCURACY= ", self.getAcurracy())
                    fileManager.appendFile("epoch: "+str(epoch)+" - Acc: "+str(self.getAcurracy()))


            for i, data in enumerate(dataGenerator._trainoader):

                if self.cudaFlag == True:
                    inputs, labels_data = data[0].cuda(), data[1].cuda()
                else:
                    inputs, labels_data = data[0], data[1]

                self.assignLabels(labels_data)
                self.total_value = 0
                self.optimizer.zero_grad()
                self.Train(inputs, 1, 1)
                self.optimizer.step()
                #scheduler.step()

                self.total_value = self.__getLossLayer().value.item()
                self.__accumulated_loss += self.total_value
                self.history_loss.append(self.total_value)


                if i % print_every == print_every - 1:
                    self.__printValues(epoch + 1, i, avg=(print_every))

            epoch+= 1
            
            end_time = time.time()

            print("epoch time: ", (end_time - start_time))

    def iterTraining(self, dataGenerator, dt_array, ricap=None):

        iters = len(dt_array)

        print_every = iters // 4
        start = time.time()
        data_iter = iter(dataGenerator._trainoader)
        
        for i in range(iters):
            
            try:
                
                data = next(data_iter)

                self.optimizer = optim.SGD(self.parameters(), lr=dt_array[i], momentum=self.momentum, weight_decay=self.weight_decay)

                if self.cudaFlag == True:
                    inputs, labels_data = data[0].cuda(), data[1].cuda()
                else:
                    inputs, labels_data = data[0], data[1]
                
                
                if ricap == None:
                    self.__doTraining(inputs=inputs, labels_data=labels_data)
                else:
                    self.__doTrainingRICAP(inputs=inputs, labels_data=labels_data, ricap=ricap)

                self.__currentEpoch = i

                if print_every > 0:

                    if i % print_every == print_every - 1:
                        
                        end_time = time.time() - start
                        self.__printValues(epoch=1, i=i, avg=print_every, end_time=end_time)
                        start = time.time()

            except StopIteration:
                data_iter = iter(dataGenerator._trainoader)

    def getLRCosine(self, total_steps, base_dt, etamin):

        array_lr = []
        self.optimizer = optim.SGD(self.parameters(), lr=base_dt, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, total_steps, eta_min=etamin)

        for i in range(total_steps):
            array_lr.append(scheduler.get_lr()[0])
            self.optimizer.step()
            scheduler.step()

        return array_lr


    def Training(self, data, labels=None, dt=0.1, p=1, full_database=False, epochs=None):

            dt_param = dt
            if type(dt) is float:
                self.optimizer = optim.SGD(self.parameters(), lr=dt, momentum=self.momentum, weight_decay=self.weight_decay)
                dt_param = None

            if full_database == False:
                self.__defaultTraining(dataGenerator=data, p=p, dt=dt_param)
            else:

                if epochs != None:
                    self.__fullDatabaseTraining(dataGenerator=data, epochs=epochs, dt=dt_param)
                else:
                    if data.type == datagen_type.DATABASE_IMAGES:
                        self.__defaultTraining(dataGenerator=data, p=p, dt=dt_param)
                    else:
                        self.__trainingRandomBatch(dataGenerator=data, p=p, dt=dt_param)

    def __defaultTraining(self, dataGenerator, p, dt=None):

            i=0

            print_every = p // 4
            start_time = time.time()
            while i < p:

                if dt is not None:
                    self.optimizer = optim.SGD(self.parameters(), lr=dt[i], momentum=self.momentum, weight_decay=self.weight_decay)

                if self.cudaFlag == True:
                    inputs, labels_data = dataGenerator.data[0].cuda(), dataGenerator.data[1].cuda()
                else:
                    inputs, labels_data = dataGenerator.data[0], dataGenerator.data[1]

                self.__doTraining(inputs=inputs, labels_data=labels_data)

                self.__currentEpoch = i

                dataGenerator.update()

                if print_every > 0:
                    if i % print_every == print_every - 1:
                        end_time = time.time() - start_time
                        self.__printValues(1, i, avg=print_every, end_time=end_time)
                        start_time = time.time()

                i=i+1

    def __trainingRandomBatch(self, dataGenerator, p, dt=None):

        i = 0
        while i < p:

            if dt is not None:
                self.optimizer = optim.SGD(self.parameters(), lr=dt[i], momentum=self.momentum, weight_decay=self.weight_decay)

            data = dataGenerator.get_random_batch()

            if self.cudaFlag == True:
                inputs, labels_data = data[0].cuda(), data[1].cuda()
            else:
                inputs, labels_data = data[0], data[1]

            self.__doTraining(inputs=inputs, labels_data=labels_data)

            self.__currentEpoch = i
            i += 1

    def __fullDatabaseTraining(self, dataGenerator, epochs, dt=None):
        epoch=1

        if dataGenerator.type == datagen_type.DATABASE_IMAGES:

            while epoch <= epochs:

                for i, data in enumerate(dataGenerator._trainoader):

                    if dt is not None:
                        self.optimizer = optim.SGD(self.parameters(), lr=dt[i], momentum=self.momentum, weight_decay=self.weight_decay)

                    if self.cudaFlag == True:
                        inputs, labels_data = data[0].cuda(), data[1].cuda()
                    else:
                        inputs, labels_data = data[0], data[1]

                    self.__doTraining(inputs=inputs, labels_data=labels_data)

                self.__currentEpoch = epoch
                epoch += 1

        elif dataGenerator.type == datagen_type.OWN_IMAGE:

            while epoch <= epochs:

                i = 0
                for data in dataGenerator.batch(dataGenerator.batch_size):

                    if dt is not None:
                        self.optimizer = optim.SGD(self.parameters(), lr=dt[i], momentum=self.momentum, weight_decay=self.weight_decay)

                    if self.cudaFlag == True:
                        inputs, labels_data = data[0].cuda(), data[1].cuda()
                    else:
                        inputs, labels_data = data[0], data[1]

                    self.__doTraining(inputs=inputs, labels_data=labels_data)

                    i += 1
                self.__currentEpoch = epoch
                epoch += 1

        else:
            print("ERROR UNKNOWN DATAGENERATOR TPYE")



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

        network = Network(newADN,cudaFlag=self.cudaFlag, momentum=self.momentum, 
            weight_decay=self.weight_decay, enable_activation=self.enable_activation, 
            enable_track_stats=self.enable_track_stats, dropout_value=self.dropout_value, 
            dropout_function=self.dropout_function, enable_last_activation=self.enable_last_activation,
            version=self.version, eps_batchnorm=self.eps_batchnorm)

        for i in range(len(self.nodes) - 1):
            layerToClone = self.nodes[i].objects[0]
            layer = network.nodes[i].objects[0]

            if layerToClone.getBias() is not None:
                layer.setBias(layerToClone.getBias().clone())

            if layerToClone.getFilter() is not None:
                layer.setFilter(layerToClone.getFilter().clone())
            
            layer.setBarchNorm(layerToClone.getBatchNormObject())

        network.total_value = self.total_value
        network.momentum = self.momentum
        network.weight_decay = self.weight_decay
        network.enable_activation = self.enable_activation
        network.enable_track_stats = self.enable_track_stats

        network.history_loss = self.history_loss[-200:]

        return network

    def generateEnergy(self, dataGen):

        accuracy = 0

        model = self.eval()
        #model = model.eval()

        with torch.no_grad():

            total = 0
            correct = 0

            for i, data in enumerate(dataGen._testloader):

                if self.cudaFlag == True:
                    inputs, labels = data[0].cuda(), data[1].cuda()
                else:
                    inputs, labels = data[0], data[1]

                if len(inputs.size()) > 4:
                    self.__getLossLayer().setCrops(inputs.shape[1])
                    inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])

                self.__getLossLayer().setEnableRicap(False)
                model.assignLabels(labels)
                model.nodes[0].objects[0].value = inputs 
                model(model.nodes[0].objects[0].value)

                linearValue = model.__getLayerProbability().value

                _, predicted = torch.max(linearValue.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total

        del model
        
        self.__accuracy = accuracy
        self.train()
    

    def getAcurracy(self):

        return self.__accuracy

    def saveModel(self, path):

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'adn': self.adn,
            'cuda': self.cudaFlag,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'enable_activation': self.enable_activation,
            'enable_track_stats': self.enable_track_stats,
            'dropout_value': self.dropout_value,
            'enable_last_activation': self.enable_last_activation,
            'version': self.version,
            'eps_batchnorm': self.eps_batchnorm
            }, path)


    def loadModel(self, path):

        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train()

    def loadParameters(self, checkpoint):
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train()

    def __printValues(self, epoch, i, avg=1, end_time=None):

        if end_time is None:
            print("[{:d}, {:d}, lr={:.10f}, Loss={:.10f}]".format(epoch, i+1, self.optimizer.param_groups[0]['lr'], self.getAverageLoss(avg)))
        else:
            print("[{:d}, {:d}, lr={:.10f}, Loss={:.10f}, Time={:.4f}]".format(epoch, i+1, self.optimizer.param_groups[0]['lr'], self.getAverageLoss(avg), end_time))
    
    def deleteParameters(self):
        
        for node in self.nodes:
            node.objects[0].deleteParam()
            del node

        del self.optimizer
        del self.nodes
        del self.history_loss

        if self.cudaFlag == True:
            torch.cuda.empty_cache()