from abc import ABC, abstractmethod
import torch

class Generator(ABC):

    def __init__(self, comp, s, source, target):

        self.Comp = comp
        self.S = s
        self.source = source
        self.target = target
        self.data = []
        self.size = None
        self.label = [0,0]


    @abstractmethod
    def generateData(self):
        pass

    def dataConv2d(self):
        self.label[0] = torch.tensor([1,0], dtype=torch.float32)
        self.label[1] = torch.tensor([0,1], dtype=torch.float32)
        self.generateData()
        self.__convertDataToPytorch()


    def dataConv3d(self):
        self.label[0] = torch.tensor([1,0], dtype=torch.float32)
        self.label[1] = torch.tensor([0,1], dtype=torch.float32)
        self.generateData()        
        self.__convertDataToPytorch(True)

    def dataNumpy(self):
        self.label[0] = "c"
        self.label[1] = "n"
        self.generateData()

    def __convertDataToPytorch(self, conv3d=False):

        for data in self.data:
            data[0] = self.numpyToTorch(data[0], conv3d)        
    
    def numpyToTorch(self, data, conv3d=False):

        data = torch.from_numpy(data).float()
        data.unsqueeze_(0)
        data.transpose_(1, 3)
        data.transpose_(2, 3)

        if conv3d == True:
            size = data.shape
            data.resize_(1, 1, size[2], size[3])
        
        return data