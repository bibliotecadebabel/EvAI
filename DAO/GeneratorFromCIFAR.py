import children.Interfaces as Inter
import children.Operations as Op
import children.net2.Network as network
from DAO.Generator import Generator

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GeneratorFromCIFAR(Generator):
    def __init__(self, comp, batchSize):
        super().__init__(comp, batchSize, "CIFAR", "folder")

        self.batchSize = batchSize
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=False, transform=self.transform)
        self.__trainoader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batchSize, shuffle=True, num_workers=0)

    def generateData(self):
        
        del self.data
    
        self.data = None

        for i, data in enumerate(self.__trainoader):
            
            inputs, labels = data

            inputs = inputs*255

            self.data = [inputs, labels]

            if i >= 0:
                break

    
    def update(self):

        self.generateData()

    def dataConv2d(self):
        self.generateData()
        self.size = [3, 32, 32]