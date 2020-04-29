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
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=False, transform=self.transform)

    def generateData(self):


        self.data = torch.utils.data.DataLoader(self.trainset, batch_size=self.batchSize, shuffle=True, num_workers=2)


    def dataConv2d(self):
        self.generateData()