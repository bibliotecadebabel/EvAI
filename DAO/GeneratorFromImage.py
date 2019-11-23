import children.Interfaces as Inter
import children.Operations as Op
import children.net2.Network as network
import numpy as np
from DAO.Generator import Generator

class GeneratorFromImage(Generator):
    def __init__(self, comp, s, cuda=True):
        super().__init__(comp, s, "btest", "folder", cuda)

        self.A = None
        self.x = None

    def generateData(self):

        batch = []
        images = []
        labels = []
        
        print('Compressing and Vectorizing input.')
        A=Op.Pool(Inter.Image2array(self.source),self.Comp)
        x=Op.Pool(Inter.Image2array(self.target),self.Comp)
        size=np.shape(x)
        print('Sampling Image')
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
        
