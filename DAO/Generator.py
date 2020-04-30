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
        self._testData = []


    @abstractmethod
    def generateData(self):
        pass

    def dataConv2d(self):

        self.label[0] = torch.tensor([0], dtype=self.dtypeLong)
        self.label[1] = torch.tensor([1], dtype=self.dtypeLong)   

        self.generateData()
        self.__convertDataToPytorch()
        self.__generateSize()
    
    def update(self):
        pass

    def dataConv3d(self):
        
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

        batch = torch.zeros(s, 3, self.data[0][0].shape[0], self.data[0][0].shape[1]).float()

        for i in range(s):
            batch[i] = self.numpyToTorch(self.data[0][i], conv3d)
        
        self.data[0] = None
        self.data[0] = batch

        self.data[1] = torch.tensor(self.data[1]).long()
    
    def numpyToTorch(self, data, conv3d=False):

        data = torch.from_numpy(data).float()
            
        data.transpose_(0, 2)
        data.transpose_(1, 2)

        if conv3d == True:
            size = data.shape
            data.resize_(1, 1, size[2], size[3])
        
        return data

    def __generateSize(self):

        self.size = self.data[0][0].shape 
