import children.Interfaces as Inter
import children.Operations as Op
import children.net2.Network as network
from DAO.Generator import Generator

class GeneratorFromCIFAR(Generator):
    def __init__(self, comp, s):
        super().__init__(comp, s, "CIFAR", "folder")
 

    def generateData(self):

        print("Generating data from ", self.source)
        '''
        print('Compressing and Vectorizing input.')
        A=Op.Pool(Inter.Image2array(self.Image),self.Comp)
        x=Op.Pool(Inter.Image2array(self.Target),self.Comp)
        size=np.shape(x)
        print('Sampling Image')
        self.Data = Op.SampleVer2((size[0],size[1]),A,self.S, "n")
        imageTarget = []
        imageTarget.append(x)
        imageTarget.append("c")
        self.Data.insert(0, imageTarget)
        self.size=size
        '''
