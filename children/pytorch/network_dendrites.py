import children.pytorch.Node as nd
import children.pytorch.layers.learnable_layers.layer_learnable as layer_learnable
import children.pytorch.layers.learnable_layers.layer_conv2d as layer_conv2d
import children.pytorch.Functions as functions
import children.pytorch.network_abstract as na
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

    def __init__(self, adn, cuda_flag=True, momentum=0.0, weight_decay=0.0, 
                enable_activation=True, enable_track_stats=True, dropout_value=0, 
                dropout_function=None, enable_last_activation=True, version=None, eps_batchnorm=None):
        nn.Module.__init__(self)
        na.NetworkAbstract.__init__(self,adn=adn, cuda=cuda_flag, momentum=momentum, weight_decay=weight_decay, 
                                enable_activaiton=enable_activation, enable_track_stats=enable_track_stats, dropout_value=dropout_value,
                                enable_last_activation=enable_last_activation)
        
        self.dropout_function = dropout_function
        self.version = version
        self.eps_batchnorm = eps_batchnorm

        if dropout_function == None:
            self.dropout_function = self.__defaultDropoutFunction
        
    
        self.__len_nodes = 0
        self.__accumulated_loss = 0
        self.__accuracy = 0

        self.__conv2d_propagate_mode = const.CONV2D_MULTIPLE_INPUTS

        if self.version == directions_version.H_VERSION or self.version == directions_version.CONVEX_VERSION:
            self.__conv2d_propagate_mode = const.CONV2D_PADDING

        self.createStructure()
        self.__currentEpoch = 0
        self.optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=self.momentum, weight_decay=self.weight_decay)
        self.eval_iter = None  
        self.acc_array = []
          
    def __defaultDropoutFunction(self, base_p, total_layers, index_learnable_layer, isPool=False):

        return base_p

    def createStructure(self):

        graph = Graphs.Graph(True)

        for i in range(len(self.adn)):

            layer_tuple = self.adn[i]

            if layer_tuple[0] != 3:
                self.__len_nodes += 1

        for i in range(self.__len_nodes):
            self.nodes.append(nd.Node())

        for i in range(self.__len_nodes):

            if i < (self.__len_nodes - 1):
                graph.add_node(i, self.nodes[i])
                graph.add_node(i + 1, self.nodes[i + 1])
                graph.add_edges(i, [i+1])


        self.nodes = list(graph.key2node.values())

        self.__assign_layers()

    def __assign_layers(self):

        #self.nodes[0].objects.append(ly.Layer(adn=self.adn[0], node=self.nodes[0]))

        indexNode = 0
        index_learnable_layer = 0

        for adn in self.adn:

            layer_tuple = adn

            if layer_tuple[0] != 3:
                
                layer = self.factory.findValue(layer_tuple, propagate_mode=self.__conv2d_propagate_mode, 
                                        enable_activation=self.enable_activation)

                layer.node = self.nodes[indexNode]
                self.nodes[indexNode].objects.append(layer)
                attributeName = "layer"+str(indexNode)
                self.set_attribute(attributeName, layer.object)
                
                #if index_learnable_layer == self.__total_layers - 2 and self.enable_last_activation == False:
                #    layer.set_enable_activation(False)

                if isinstance(layer, layer_learnable.LearnableLayer):
                    
                    if self.cuda_flag == True:
                        tensor_h = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True).cuda())
                    else:
                        tensor_h = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

                    layer_pool = False

                    if layer_tuple[0] == 0 and len(layer_tuple) > 5:
                        layer_pool = True

                    dropout_value = self.dropout_function(self.dropout_value, self.__len_nodes-1, index_learnable_layer, layer_pool)

                    layer.set_dropout_value(dropout_value)
                    conv2d_dropout = torch.nn.Dropout2d(p=dropout_value)

                    if self.cuda_flag == True:
                        conv2d_dropout = conv2d_dropout.cuda()                    

                    layer.set_dropout(conv2d_dropout)
                    attributeName_dropout = "layer_dropout"+str(indexNode)
                    self.set_attribute(attributeName_dropout, layer.get_dropout())
                    
                    layer.tensor_h = tensor_h
                    attributeName_h = "layer_h"+str(indexNode)
                    self.set_attribute(attributeName_h, layer.tensor_h)

                    index_learnable_layer += 1

                    if isinstance(layer, layer_conv2d.Conv2dLayer):

                        if self.eps_batchnorm == None:
                            conv2d_batchnorm = torch.nn.BatchNorm2d(layer_tuple[2], track_running_stats=self.enable_track_stats)
                        else:
                            conv2d_batchnorm = torch.nn.BatchNorm2d(layer_tuple[2], track_running_stats=self.enable_track_stats, eps=self.eps_batchnorm)

                        if self.cuda_flag == True:
                            conv2d_batchnorm = conv2d_batchnorm.cuda()
                        
                        layer.set_batch_norm_object(conv2d_batchnorm)
                        attributeName_batch = "layer_batchnorm"+str(indexNode)
                        self.set_attribute(attributeName_batch, layer.get_batch_norm_object())

                        if len(layer_tuple) > 5:
                            conv2d_pool = torch.nn.MaxPool2d((layer_tuple[5], layer_tuple[5]), stride=None, ceil_mode=True)

                            if self.cuda_flag == True:
                                conv2d_pool = conv2d_pool.cuda()

                            layer.set_pool(conv2d_pool)
                            attributeName_pool = "layer_pool"+str(indexNode)
                            self.set_attribute(attributeName_pool, layer.get_pool())

                indexNode += 1

            else:
                input_node = layer_tuple[1]+1
                target_node = layer_tuple[2]+1
                self.nodes[target_node].objects[0].connected_layers.append(self.nodes[input_node].objects[0])

    def predict(self, image, label):

        labelTensor = na.tensorFactory.createTensor(body=[label.item()], cuda=self.cuda_flag, requiresGrad=False)
        self.assign_labels(labelTensor)

        value = image.view(1, 3, image.shape[1], image.shape[2])

        if self.cuda_flag == True:
            value = value.cuda()

        self.nodes[0].objects[0].value = value
        self(self.nodes[0].objects[0].value)

        return self.get_prediction_result()


    def get_prediction_result(self):

        value = self.get_linear_layer().value

        sc = value[0][0]
        sn = value[0][1]

        p = torch.exp(sc)/(torch.exp(sc) + torch.exp(sn))

        return p.item()

    def __init_training(self, data):

        self.nodes[0].objects[0].value = data
        self.set_grad_flag(True)

        self(data)
        self.__start_back_propagation()
        self.set_grad_flag(False)

    def __default_training(self, inputs, labels_data):

        self.get_loss_layer().set_ricap(None)
        self.assign_labels(labels_data)
        self.total_value = 0
        self.optimizer.zero_grad()
        self.__init_training(inputs)
        self.optimizer.step()

        self.total_value = self.get_loss_layer().value.item()
        self.__accumulated_loss += self.total_value

    
    def __training_with_ricap(self, inputs, labels_data, ricap : Augmentation.Ricap):

        self.get_loss_layer().set_enable_ricap(True)
        self.assign_labels(labels_data)
        self.total_value = 0
        self.optimizer.zero_grad()

        patched_images = ricap.doRicap(inputs=inputs, target=labels_data, cuda=self.cuda_flag)        
        self.get_loss_layer().set_ricap(ricap)
        self.__init_training(patched_images)
        self.optimizer.step()

        self.total_value = self.get_loss_layer().value.item()
        self.__accumulated_loss += self.total_value

    def training_cosine_dt(self, dataGenerator, max_dt, min_dt, epochs, restart_dt=1, show_accuarcy=False, fileManager=None):

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
                    self.generate_accuracy(dataGen=dataGenerator)
                    print("ACCURACY= ", self.get_accuracy())
                    fileManager.appendFile("epoch: "+str(epoch)+" - Acc: "+str(self.get_accuracy()))


            for i, data in enumerate(dataGenerator._trainoader):

                if self.cuda_flag == True:
                    inputs, labels_data = data[0].cuda(), data[1].cuda()
                else:
                    inputs, labels_data = data[0], data[1]

                self.assign_labels(labels_data)
                self.total_value = 0
                self.optimizer.zero_grad()
                self.__init_training(inputs)
                self.optimizer.step()
                #scheduler.step()

                self.total_value = self.get_loss_layer().value.item()
                self.__accumulated_loss += self.total_value
                self.history_loss.append(self.total_value)


                if i % print_every == print_every - 1:
                    self.__print_training_values(epoch + 1, i, avg=(print_every))

            epoch+= 1
            
            end_time = time.time()

            print("epoch time: ", (end_time - start_time))

    def training_custom_dt(self, dataGenerator, dt_array, ricap=None, evalLoss=False):

        try:
            print("Using eva loss: ", evalLoss)
            print("ricap: ", ricap)
            save_acc = False
            current_epoch = 0
            iters = len(dt_array)

            print_every = iters // 10
            start = time.time()
            data_iter = iter(dataGenerator._trainoader)

            if evalLoss == True:
                self.eval_iter = iter(dataGenerator._evalloader)
            
            i = 0

            while i < iters:
                
                try:
                                        
                    data = next(data_iter)

                    self.optimizer = optim.SGD(self.parameters(), lr=dt_array[i], momentum=self.momentum, weight_decay=self.weight_decay)

                    if self.cuda_flag == True:
                        inputs, labels_data = data[0].cuda(), data[1].cuda()
                    else:
                        inputs, labels_data = data[0], data[1]
                    
                    
                    if ricap == None:
                        self.__default_training(inputs=inputs, labels_data=labels_data)
                    else:
                        self.__training_with_ricap(inputs=inputs, labels_data=labels_data, ricap=ricap)

                    self.__currentEpoch = i
                    
                    if evalLoss == False:
                        self.history_loss.append(self.total_value)
                    else:
                        self.__generate_eval_loss(dataGenerator)
                        
                    if print_every > 0:

                        if i % print_every == print_every - 1:
                            
                            end_time = time.time() - start
                            self.__print_training_values(epoch=1, i=i, avg=print_every, end_time=end_time)
                            start = time.time()
                    i+= 1

                except StopIteration:
                    current_epoch += 1
                    save_acc = True
                    data_iter = iter(dataGenerator._trainoader)

        except:
            print("ERROR TRAINING")
            print("DNA: ", self.adn)
            raise

    def __generate_eval_loss(self, dataGenerator):

        with torch.no_grad():
            stop = False

            eval_data = None

            while stop == False:

                try:
                    eval_data = next(self.eval_iter)
                    stop = True
                except StopIteration:
                    self.eval_iter =  iter(dataGenerator._evalloader)
                
            model = self.eval()

            if self.cuda_flag == True:
                inputs, labels = eval_data[0].cuda(), eval_data[1].cuda()
            else:
                inputs, labels = eval_data[0], eval_data[1]
            
            if len(inputs.size()) > 4:
                self.get_loss_layer().set_crops(inputs.shape[1])
                inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])

            self.get_loss_layer().set_enable_ricap(False)

            model.assign_labels(labels)
            model.nodes[0].objects[0].value = inputs 
            model(model.nodes[0].objects[0].value)
            eval_loss = model.get_loss_layer().value.item()
            self.history_loss.append(eval_loss)
            self.train()

    def forward(self, x):
        self.__start_propagation()

    def __start_propagation(self):

        functions.Propagation(self.get_loss_layer())

    def __start_back_propagation(self):

        self.get_loss_layer().value.backward()

    def clone(self):

        newObjects = []
        newADN = tuple(list(self.adn))

        network = Network(newADN,cuda_flag=self.cuda_flag, momentum=self.momentum, 
            weight_decay=self.weight_decay, enable_activation=self.enable_activation, 
            enable_track_stats=self.enable_track_stats, dropout_value=self.dropout_value, 
            dropout_function=self.dropout_function, enable_last_activation=self.enable_last_activation,
            version=self.version, eps_batchnorm=self.eps_batchnorm)

        for i in range(len(self.nodes) - 1):
            layerToClone = self.nodes[i].objects[0]
            layer = network.nodes[i].objects[0]

            if layerToClone.get_bias() is not None:
                layer.set_bias(layerToClone.get_bias().clone())

            if layerToClone.get_filters() is not None:
                layer.set_filters(layerToClone.get_filters().clone())
            
            if layerToClone.tensor_h is not None:
                layer.tensor_h.data = layerToClone.tensor_h.data.clone()
                
            layer.set_batch_norm(layerToClone.get_batch_norm_object())

        network.total_value = self.total_value
        network.momentum = self.momentum
        network.weight_decay = self.weight_decay
        network.enable_activation = self.enable_activation
        network.enable_track_stats = self.enable_track_stats

        network.history_loss = self.history_loss[-200:]

        return network

    def generate_accuracy(self, dataGen):

        accuracy = 0

        model = self.eval()
        #model = model.eval()

        with torch.no_grad():

            total = 0
            correct = 0

            for i, data in enumerate(dataGen._testloader):

                if self.cuda_flag == True:
                    inputs, labels = data[0].cuda(), data[1].cuda()
                else:
                    inputs, labels = data[0], data[1]

                if len(inputs.size()) > 4:
                    self.get_loss_layer().set_crops(inputs.shape[1])
                    inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])

                self.get_loss_layer().set_enable_ricap(False)
                model.assign_labels(labels)
                model.nodes[0].objects[0].value = inputs 
                model(model.nodes[0].objects[0].value)

                linearValue = model.get_linear_layer().value

                _, predicted = torch.max(linearValue.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total

        del model
        
        self.__accuracy = accuracy
        self.train()
    

    def get_accuracy(self):

        return self.__accuracy

    def save_model(self, path):

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'adn': self.adn,
            'cuda': self.cuda_flag,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'enable_activation': self.enable_activation,
            'enable_track_stats': self.enable_track_stats,
            'dropout_value': self.dropout_value,
            'enable_last_activation': self.enable_last_activation,
            'version': self.version,
            'eps_batchnorm': self.eps_batchnorm
            }, path)


    def load_model(self, path):

        checkpoint = torch.load(path)
        self.load_parameters(checkpoint)

    def load_parameters(self, checkpoint):
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train()

    def __print_training_values(self, epoch, i, avg=1, end_time=None):

        if end_time is None:
            print("[{:d}, {:d}, lr={:.10f}, Loss={:.10f}]".format(epoch, i+1, self.optimizer.param_groups[0]['lr'], self.get_average_loss(avg)))
        else:
            print("[{:d}, {:d}, lr={:.10f}, Loss={:.10f}, Time={:.4f}]".format(epoch, i+1, self.optimizer.param_groups[0]['lr'], self.get_average_loss(avg), end_time))
    
    def __printAcc(self, epoch, acc):
        print("[Epoch={:d}, lr={:.10f}, Acc={:.4f}]".format(epoch, self.optimizer.param_groups[0]['lr'], acc))
    

    def delete_parameters(self):
        
        for node in self.nodes:
            node.objects[0].deleteParam()
            del node

        del self.optimizer
        del self.nodes
        del self.history_loss

        if self.cuda_flag == True:
            torch.cuda.empty_cache()