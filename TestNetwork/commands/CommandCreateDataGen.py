from DAO import GeneratorFromImage

class CommandCreateDataGen():

    def __init__(self, cuda=False):
        self.__dataGen = None
        self.__cuda = cuda

    def execute(self, compression, amount_images):
        self.__dataGen = GeneratorFromImage.GeneratorFromImage(compression, amount_images, cuda=self.__cuda)
        self.__dataGen.dataConv2d()

    
    def returnParam(self):
        return self.__dataGen
    