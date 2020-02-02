from abc import ABC, abstractmethod
import torch

class Generator(ABC):

    def __init__(self, comp, s, source, target, cuda=True):

        self.Comp = comp
        self.S = s
        self.source = source
        self.target = target
        self.data = []
        self.size = None
        self.label = [0,0]
        self.dtypeFloat = torch.float32
        self.dtypeLong = torch.long
        self.cuda = cuda


    @abstractmethod
    def generateData(self):
        pass

    def dataConv2d(self):

        if self.cuda == True:
            self.label[0] = torch.tensor([0], dtype=self.dtypeLong).cuda()
            self.label[1] = torch.tensor([1], dtype=self.dtypeLong).cuda()
        else:
            self.label[0] = torch.tensor([0], dtype=self.dtypeLong)
            self.label[1] = torch.tensor([1], dtype=self.dtypeLong)   

        self.generateData()
        self.__convertDataToPytorch()
        self.__generateSize()

    def dataConv3d(self):
        
        if self.cuda == True:
            self.label[0] = torch.tensor([0], dtype=self.dtypeLong).cuda()
            self.label[1] = torch.tensor([1], dtype=self.dtypeLong).cuda()
        else:
            self.label[0] = torch.tensor([0], dtype=self.dtypeLong)
            self.label[1] = torch.tensor([1], dtype=self.dtypeLong)

        self.generateData()        
        self.__convertDataToPytorch(True)
        self.__generateSize()

    def dataNumpy(self):
        self.label[0] = "c"
        self.label[1] = "n"
        self.generateData()
        self.__generateSize()

    def __convertDataToPytorch(self, conv3d=False):
        
        s = len(self.data[0])

        if conv3d == True:
            batch = self.__generateTensorConv3d(s)
        else:
            batch = self.__generateTensorConv2d(s)

        for i in range(s):
            batch[i] = self.numpyToTorch(self.data[0][i], conv3d)
        
        self.data[0] = None
        self.data[0] = batch

        if self.cuda == True:
            self.data[1] = torch.tensor(self.data[1]).long().cuda()
        else:
            self.data[1] = torch.tensor(self.data[1]).long()
    
    def __generateTensorConv2d(self, length):

        if self.cuda == True:
            batch = torch.zeros(length, 3, self.data[0][0].shape[0], self.data[0][0].shape[1]).float().cuda()
        else:
            batch = torch.zeros(length, 3, self.data[0][0].shape[0], self.data[0][0].shape[1]).float()
        
        return batch
    
    def __generateTensorConv3d(self, length):

        if self.cuda == True:
            batch = torch.zeros(length, 3, 1, self.data[0][0].shape[0], self.data[0][0].shape[1]).float().cuda()
        else:
            batch = torch.zeros(length, 3, 1, self.data[0][0].shape[0], self.data[0][0].shape[1]).float()
        
        return batch

    def numpyToTorch(self, data, conv3d=False):

        if self.cuda == True:
            data = torch.from_numpy(data).float().cuda()
        else:
            data = torch.from_numpy(data).float()
            
        data.transpose_(0, 2)
        data.transpose_(1, 2)

        if conv3d == True:
            size = data.shape
            data.resize_(3, 1, size[1], size[2])
        return data

    def __generateSize(self):

        self.size = self.data[0][0].shape 
