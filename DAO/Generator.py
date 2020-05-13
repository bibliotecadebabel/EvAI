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
        self.trainloader = None
        self.type = None
        self.target_tensor = None


    @abstractmethod
    def generateData(self):
        pass

    def generateTrainLoader(self):
        pass
    
    def batch(self, n=1):
        l = len(self.trainloader[0])
        for ndx in range(0, l, n):
            value =  [self.trainloader[0][ndx:min(ndx + n, l)], self.trainloader[1][ndx:min(ndx + n, l)]]
            
            value[0] = torch.cat((self.target_tensor[0].clone(), value[0]), dim=0)
            value[1] = torch.cat((self.target_tensor[1].clone(), value[1]), dim=0)

            yield value

    def dataConv2d(self):

        self.label[0] = torch.tensor([0], dtype=self.dtypeLong)
        self.label[1] = torch.tensor([1], dtype=self.dtypeLong)   

        self.generateData()
        self.__convertDataToPytorch()
        self.__generateSize()
    
    def generateDataBase(self):

        self.label[0] = torch.tensor([0], dtype=self.dtypeLong)
        self.label[1] = torch.tensor([1], dtype=self.dtypeLong) 
        
        self.data = None
        self.trainloader = None

        self.generateTrainLoader() 
        self.__convertDatabaseToPytorch()
        self.__generateSize_db()
    
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
    
    def __convertDatabaseToPytorch(self):
        s = len(self.trainloader[0])
        s_target = len(self.target_tensor[0])

        print("database size=", s)

        batch = torch.zeros(s, 3, self.trainloader[0][0].shape[0], self.trainloader[0][0].shape[1]).float()
        target = torch.zeros(s_target, 3, self.target_tensor[0][0].shape[0], self.target_tensor[0][0].shape[1]).float()


        for i in range(s):
            batch[i] = self.numpyToTorch(self.trainloader[0][i], False)
        
        for j in range(s_target):
            target[j] = self.numpyToTorch(self.target_tensor[0][j], False)
        
        self.trainloader[0] = None
        self.trainloader[0] = batch

        self.target_tensor[0] = None
        self.target_tensor[0] = target

        self.trainloader[1] = torch.tensor(self.trainloader[1]).long()
        self.target_tensor[1] = torch.tensor(self.target_tensor[1]).long()
    
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
    
    def __generateSize_db(self):

        self.size = self.trainloader[0][0].shape
