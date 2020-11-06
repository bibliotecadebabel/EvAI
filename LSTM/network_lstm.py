import torch as torch
import torch.nn as nn
import torch.tensor as tensor

import torch.optim as optim
import LSTM.memory_unit as memory_unit
import Factory.TensorFactory as TensorFactory


class NetworkLSTM(nn.Module):

    def __init__(self, observation_size, inChannels, outChannels, kernelSize, cuda_flag=True):
        super(NetworkLSTM, self).__init__()

        self.cuda_flag = cuda_flag
        self.lenModules = observation_size-1
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.total_value = 0
        self.loss_history = []
        self.units = []
        self.modulesXT = []

        self.__createStructure()

    def set_attribute(self, name, value):
        setattr(self,name,value)

    def __get_attribute(self, name):

        attribute = None
        try:
            attribute = getattr(self, name)
        except AttributeError:
            pass

        return attribute

    def delete_attribute(self, name):
        try:
            delattr(self, name)
        except AttributeError:
            pass
    
    def __createStructure(self):
        
        for i in range(self.lenModules):

            inchannels = self.inChannels
            inChannels_candidate = inchannels

            if i > 0:
                
                #inchannels += 2 #version 1
                inchannels += self.inChannels * 2 #version 2
                inChannels_candidate = inChannels_candidate*2
            
            internal = memory_unit.MemoryUnit(kernelSize=self.kernelSize, inChannels=inchannels, inChannels_candidate=inChannels_candidate, outChannels=self.outChannels, cuda_flag=self.cuda_flag)
            self.units.append(internal)
            
            attr_1 = "internal_"+str(i)+"_convFT"
            attr_2 = "internal_"+str(i)+"_convIT"
            attr_3 = "internal_"+str(i)+"_convCND"
            attr_4 = "internal_"+str(i)+"_convOT"

            self.set_attribute(attr_1, internal.convFt)
            self.set_attribute(attr_2, internal.convIt)
            self.set_attribute(attr_3, internal.convCand)
            self.set_attribute(attr_4, internal.convOt)
    
    def Train(self, dataElement, observations):
        
        self.set_grad_flag(True)
        self(dataElement)
        self.__generate_loss(observations)
        self.__doBackward()
        self.set_grad_flag(False)

    def __createModulesXT(self, data):
        
        for indexModule in range(self.lenModules):
            self.modulesXT.append(self.__getInputModule(indexModule, data))

        self.modulesXT.append(self.__getInputModule(self.lenModules, data))
    
    def __createModulesXTPredict(self, data, length):
        
        for indexModule in range(length):
            self.modulesXT.append(self.__getInputModule(indexModule, data))

    def Training(self, data, observations, dt=0.1, p=1):
        
        self.modulesXT = []
        self.optimizer = optim.SGD(self.parameters(), lr=dt, momentum=0)
        self.__createModulesXT(data)
        i = 0
        print_every = p // 4
        while i < p:
            self.energy = 0
            self.optimizer.zero_grad()
            self.Train(data, observations)
            self.optimizer.step()

            
            if i % print_every == print_every - 1:
                print("L=", self.energy, "i=", i+1)
            
            i += 1
        

    def __getInputModule(self, moduleIndex, wordsTensor):
        batch = wordsTensor.shape[0]
        wordValue = wordsTensor.shape[2]
        shape = wordsTensor.shape

        if len(shape) > 3:
            value = TensorFactory.createTensorZeros(tupleShape=(shape[0], shape[2], shape[3]), cuda=self.cuda_flag, requiresGrad=False)
        else:
            value = TensorFactory.createTensorZeros(tupleShape=(shape[0], 1, shape[2]), cuda=self.cuda_flag, requiresGrad=False)

        i = 0
        for word in wordsTensor:
            #print("word: ", word.size())
            letter = word[moduleIndex]
            #print("letter: ", letter.size())
            value[i] = letter.clone()
            #print("value[i]: ", value[i].size())
            i += 1
        
        return value
            

    def __generate_loss(self, observations):
        

        weights = []
        for observation in observations:
            weights.append([[observation.weight]])
        
        weigth_tensor = torch.tensor(weights, dtype=torch.float32, requires_grad=False).cuda()
        indexModule = 0
        energy = 0

        ## MSELoss
        for module in self.units:
            
            value = module.ht - self.modulesXT[indexModule+1]
            value = torch.reshape(value, (value.shape[0], 1, value.shape[1]*value.shape[2]))
            value = torch.mul(value, value)
            value = torch.bmm(weigth_tensor, value)
            energy += value
            indexModule += 1
        
        self.energy = torch.div(energy, self.lenModules+1).sum()

        # multiplicar por peso de la palabra (cantidad de particulas)
        #self.energy = self.energy

    def forward(self, wordsTensor):
        last_ht = None
        last_ct = None
        moduleIndex = 0
        for module in self.units:
            xt = self.modulesXT[moduleIndex]
            module.compute(xt=xt, last_ht=last_ht, last_ct=last_ct)
            
            last_ht = module.ht
            last_ct = module.ct

            moduleIndex += 1

    def __doBackward(self):
        self.energy.backward()

    def set_grad_flag(self, flag):

        for module in self.units:

            module.set_grad_flag(flag)
    
    def predict(self, x):
    
        last_ht = None
        last_ct = None

        modules = x.shape[1]

        self.modulesXT = []
        self.__createModulesXTPredict(x, modules)

        for moduleIndex in range(modules):
            xt = self.modulesXT[moduleIndex]
            module = self.units[moduleIndex]
            module.compute(xt=xt, last_ht=last_ht, last_ct=last_ct)
            
            last_ht = module.ht
            last_ct = module.ct
        
        ht = self.units[modules-1].ht
        
        return ht
            
