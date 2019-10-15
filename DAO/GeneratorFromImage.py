import children.Interfaces as Inter
import children.Operations as Op
import children.net2.Network as network
import numpy as np
from DAO.Generator import Generator

class GeneratorFromImage(Generator):
    def __init__(self, comp, s):
        super().__init__(comp, s, "btest", "folder")

        self.A = None
        self.x = None

    def generateData(self):

        data = []
        
        print('Compressing and Vectorizing input.')
        A=Op.Pool(Inter.Image2array(self.source),self.Comp)
        x=Op.Pool(Inter.Image2array(self.target),self.Comp)
        size=np.shape(x)
        print('Sampling Image')
        data = Op.SampleVer2((size[0],size[1]),A,self.S, self.label[1])
        imageTarget = []
        imageTarget.append(x)
        imageTarget.append(self.label[0])

        data.insert(0, imageTarget)
        
        self.A = A
        self.x = x
        self.data = data
        
