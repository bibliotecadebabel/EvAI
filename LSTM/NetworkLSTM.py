import torch as torch
import torch.nn as nn
import torch.tensor as tensor

import torch.optim as optim
import LSTM.InternalModule as InternalModule
import LSTM.InternalModuleVariant as InternalModuleVariant
import Factory.TensorFactory as TensorFactory


class NetworkLSTM(nn.Module):

    def __init__(self, max_letters, inChannels, outChannels, kernelSize, cudaFlag=True):
        super(NetworkLSTM, self).__init__()

        self.cudaFlag = cudaFlag
        self.lenModules = max_letters-1
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.total_value = 0
        self.history_loss = []
        self.internalModules = []
        self.modulesXT = []

        self.__createStructure()

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
    
    def __createStructure(self):
        
        for i in range(self.lenModules):

            inchannels = self.inChannels

            if i > 0:
                inchannels += 2
            
            internal = InternalModuleVariant.InternalModuleVariant(kernelSize=self.kernelSize, inChannels=inchannels, outChannels=self.outChannels, cudaFlag=self.cudaFlag)
            self.internalModules.append(internal)
            
            attr_1 = "internal_"+str(i)+"_convFT"
            attr_2 = "internal_"+str(i)+"_convIT"
            attr_3 = "internal_"+str(i)+"_convCND"
            attr_4 = "internal_"+str(i)+"_convOT"

            self.setAttribute(attr_1, internal.convFt)
            self.setAttribute(attr_2, internal.convIt)
            self.setAttribute(attr_3, internal.convCand)
            self.setAttribute(attr_4, internal.convOt)
    
    def Train(self, dataElement):
    
        self.updateGradFlag(True)
        self(dataElement)
        self.__generateEnergy()
        self.__doBackward()
        self.updateGradFlag(False)

        #self.total_value += ((self.__getLossLayer().value)).item()

    def __createModulesXT(self, data):
        
        for indexModule in range(self.lenModules):
            self.modulesXT.append(self.__getInputModule(indexModule, data))

        self.modulesXT.append(self.__getInputModule(self.lenModules, data))
    
    def __createModulesXTPredict(self, data, length):
        
        for indexModule in range(length):
            self.modulesXT.append(self.__getInputModule(indexModule, data))

    def Training(self, data, dt=0.1, p=1):
        
        self.optimizer = optim.SGD(self.parameters(), lr=dt, momentum=0)
        self.__createModulesXT(data)
        i = 0
        
        while i < p:
            self.energy = 0
            self.optimizer.zero_grad()
            self.Train(data)
            self.optimizer.step()

            
            if i % 10 == 0:
                print("L=", self.energy, "i=", i)
            
            i += 1
        

    def __getInputModule(self, moduleIndex, wordsTensor):
        batch = wordsTensor.shape[0]
        wordValue = wordsTensor.shape[2]
        value = TensorFactory.createTensorZeros(tupleShape=(batch, 1, wordValue), cuda=self.cudaFlag, requiresGrad=False)
        
        i = 0
        for word in wordsTensor:
            letter = word[moduleIndex]
            value[i][0] = letter.clone()
            i += 1
        
        return value
            

    def __generateEnergy(self):

        indexModule = 0
        energy = 0
        for module in self.internalModules:
            value = module.ht - self.modulesXT[indexModule+1]
            value = torch.mul(value, value)
            energy += value
            indexModule += 1
        
        self.energy = torch.div(energy, self.lenModules+1).sum()
        self.energy = self.energy*1000

    def forward(self, wordsTensor):
        last_ht = None
        last_ct = None
        moduleIndex = 0
        for module in self.internalModules:
            xt = self.modulesXT[moduleIndex]
            module.compute(xt=xt, last_ht=last_ht, last_ct=last_ct)
            
            last_ht = module.ht
            last_ct = module.ct

            moduleIndex += 1

    def __doBackward(self):
        self.energy.backward()

    def updateGradFlag(self, flag):

        for module in self.internalModules:

            module.updateGradFlag(flag)
    
    def predict(self, x):
    
        indexModule = 0
        last_ht = None
        last_ct = None

        modules = x.shape[1]

        self.modulesXT = []
        self.__createModulesXTPredict(x, modules)

        for moduleIndex in range(modules):
            xt = self.modulesXT[moduleIndex]
            module = self.internalModules[moduleIndex]
            module.compute(xt=xt, last_ht=last_ht, last_ct=last_ct)
            
            last_ht = module.ht
            last_ct = module.ct
        
        ht = self.internalModules[modules-1].ht

        index = torch.argmax(ht, dim=2)

        return index.item()

            
