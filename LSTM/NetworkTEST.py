import torch as torch
import torch.nn as nn
import torch.tensor as tensor

import torch.optim as optim
import LSTM.InternalModule as InternalModule
import Factory.TensorFactory as TensorFactory

class MyLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(MyLSTM,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(input_dim,hidden_dim)
        self.LNN = nn.Linear(hidden_dim,input_dim)
    
    def Train(self, dataElement):
    
        self.updateGradFlag(True)
        self(dataElement)
        self.__generateEnergy()
        self.__doBackward()
        self.updateGradFlag(False)

    def Training(self, data, dt=0.1, p=1):
        
        self.optimizer = optim.SGD(self.parameters(), lr=dt, momentum=0)
        self.backwardTensor = TensorFactory.createTensorOnes(tupleShape=(data.shape[0], 1, data.shape[2]), cuda=self.cudaFlag,requiresGrad=False)
        self.__createModulesXT(data)
        i = 0
        
        while i < p:
            self.total_value = 0
            self.optimizer.zero_grad()
            self.Train(data)
            self.optimizer.step()
            i += 1

    def forward(self,inp,hc):

        output,_= self.LSTM(inp,hc)
        return self.LNN(output)