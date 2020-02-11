import torch
import children.pytorch.Layer as ly
import children.pytorch.Functions as functions
import Factory.AbstractFactory as AbstractFactory

class LayerGenerator(AbstractFactory.FactoryClass):

    def __init__( self, cuda=True ):
        super().__init__()
        
        self.__cuda = cuda

    def createDictionary(self):

        self.dictionary[0] = self.__createConv3d
        self.dictionary[1] = self.__createLinear
        self.dictionary[2] = self.__createCrossEntropyLoss

    def findValue(self, tupleBody):
        key = tupleBody[0]

        value = self.dictionary[key]

        return value(tupleBody)

    def __createConv2d(self, tupleBody):

        layer = torch.nn.Conv2d(tupleBody[1], tupleBody[2], (tupleBody[3], tupleBody[4]))
        #valueLayerConv2d = torch.rand(1, tupleBody[2], 1, 1, dtype=torch.float32, requires_grad=True)
        
        self.__verifyCuda(layer)
        #self.__verifyCuda(valueLayerConv2d)
        
        value = ly.Layer(objectTorch=layer, propagate=functions.conv2d_propagate, value=None, adn=tupleBody, cudaFlag=self.__cuda)
        
        return value
    
    def __createConv3d(self, tupleBody):

        layer = torch.nn.Conv3d(tupleBody[1], tupleBody[2], (tupleBody[3], tupleBody[4], tupleBody[5]))
        torch.nn.init.uniform_(layer.weight, a=0.1, b=0.3)
        torch.nn.init.uniform_(layer.bias, a=0.1, b=0.3)
        #torch.nn.init.constant_(layer.bias, 0)
        #valueLayerConv2d = torch.rand(1, tupleBody[2], 1, 1, dtype=torch.float32, requires_grad=True)
        
        self.__verifyCuda(layer)
        #self.__verifyCuda(valueLayerConv2d)
        
        value = ly.Layer(objectTorch=layer, propagate=functions.conv3d_propagate, value=None, adn=tupleBody, cudaFlag=self.__cuda)
        
        return value

    def __createLinear(self, tupleBody):
        
        layer = torch.nn.Linear(tupleBody[1], tupleBody[2])
        torch.nn.init.uniform_(layer.weight, a=0.1, b=0.3)
        torch.nn.init.uniform_(layer.bias, a=0.1, b=0.3)
        #torch.nn.init.constant_(layer.bias, 0)
        #valueLayerLinear = torch.rand(2, dtype=torch.float32, requires_grad=True)

        self.__verifyCuda(layer)
        #self.__verifyCuda(valueLayerLinear)
        
        value = ly.Layer(objectTorch=layer, adn=tupleBody,propagate=functions.linear_propagate, value=None, cudaFlag=self.__cuda)

        return value

    def __createCrossEntropyLoss(self, tupleBody):

        layer = torch.nn.CrossEntropyLoss()
        
        self.__verifyCuda(layer)

        value = ly.Layer(objectTorch=layer, adn=tupleBody, propagate=functions.MSEloss_propagate, cudaFlag=self.__cuda)

        return value


    def __verifyCuda(self, layer):

        if self.__cuda == True:
            layer.cuda()