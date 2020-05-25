import children.Interfaces as Inter
import children.Operations as Op

import const.datagenerator_type as datagen_type
from DAO.Generator import Generator

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GeneratorFromCIFAR(Generator):
    def __init__(self, comp, batchSize, cuda=False, threads=0):
        super().__init__(comp, batchSize, "CIFAR", "folder", cuda=cuda)

        self.batchSize = batchSize
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=self.transform)
        self.testSet = torchvision.datasets.CIFAR10(root='./cifar', train=False, download=False, transform=self.transform)
        print("threads = ", threads)
        self._trainoader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batchSize, shuffle=True, num_workers=threads)
        self._testloader = torch.utils.data.DataLoader(self.testSet, batch_size=32, shuffle=False, num_workers=threads)
        self.type = datagen_type.DATABASE_IMAGES
        self.total_steps = len(self._trainoader)

    def generateData(self):

        del self.data

        self.data = None

        for i, data in enumerate(self._trainoader):

            inputs, labels = data

            self.data = [inputs, labels]

            if i >= 0:
                break


    def getRandomSample(self, i):
        pass

    def __generateTestData(self):

        pass


    def update(self):

        self.generateData()

    def dataConv2d(self):
        self.generateData()
        self.size = [3, 32, 32]
