from DAO import GeneratorFromImage, GeneratorFromCIFAR

class CommandCreateDataGen():

    def __init__(self, cuda=False):
        self.__dataGen = None
        self.__cuda = cuda

    def execute(self, compression, batchSize, source='default', threads=0, dataAugmentation=False, transformCompose=None):
        
        if source == 'cifar':
            self.__dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(compression, batchSize, cuda=self.__cuda, 
                            threads=threads, dataAugmentation=dataAugmentation, transforms_mode=transformCompose)
        else:
            self.__dataGen = GeneratorFromImage.GeneratorFromImage(compression, batchSize, cuda=self.__cuda)

        self.__dataGen.dataConv2d()

    
    def returnParam(self):
        return self.__dataGen
    