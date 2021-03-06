import children.Interfaces as Inter
import children.Operations as Op
import numpy as np
from DAO.Generator import Generator
from DAO.Generator import torch
import random
import const.datagenerator_type as datagen_type

class GeneratorFromImage(Generator):
    def __init__(self, comp, batch_size,database_size=2000,cuda=False):
        super().__init__(comp, batch_size, "btest", "folder", cuda)

        self.A = None
        self.x = None
        self.type = datagen_type.OWN_IMAGE
        self.batch_size = batch_size
        self.databaseSize = database_size

    def generateData(self):

        batch = []
        images = []
        labels = []
        
        A=Op.Pool(Inter.Image2array(self.source),self.Comp)
        x=Op.Pool(Inter.Image2array(self.target),self.Comp)
        size=np.shape(x)
        images = Op.SampleVer3((size[0],size[1]),A,self.S)
        #images.insert(0, x)

        copy_x = self.S // 4 

        for _ in range(copy_x):
            images.insert(0, np.copy(x))
            labels.append(self.label[0])

        for _ in range(self.S):
            labels.append(self.label[1])

        batch.append(images)
        batch.append(labels)
        
        self.A = A
        self.x = x
        self.data = batch
    
    def generateTrainLoader(self):
        
        print("generating database")
        database = []
        images = []
        labels = []

        target = []
        image_target = []
        label_target = []
        
        A=Op.Pool(Inter.Image2array(self.source),self.Comp)
        x=Op.Pool(Inter.Image2array(self.target),self.Comp)
        size=np.shape(x)
        images = Op.SampleVer3((size[0],size[1]),A,self.databaseSize)
        #images.insert(0, x)

        target_ammount = self.batch_size // 4

        for _ in range(target_ammount):
            image_target.append(np.copy(x))
            label_target.append(self.label[0])

        for _ in range(self.databaseSize):
            labels.append(self.label[1])

        database.append(images)
        database.append(labels)

        target.append(image_target)
        target.append(label_target)
        
        
        self.A = A
        self.x = x


        self.trainloader = database
        self.target_tensor = target
    
    def get_random_batch(self):
        
        last_index = self.databaseSize - self.batch_size
        
        min_range = random.randint(0, last_index)
        max_range = min_range + self.batch_size

        batch = [self.trainloader[0][min_range:max_range], self.trainloader[1][min_range:max_range]]

        batch[0] = torch.cat((self.target_tensor[0].clone(), batch[0]), dim=0)
        batch[1] = torch.cat((self.target_tensor[1].clone(), batch[1]), dim=0)

        return batch



    
        
        
