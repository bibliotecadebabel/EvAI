import torch
import children.pytorch.Layer as ly
import children.pytorch.Functions as functions

class LayerGenerator():

    def __init__( self, cuda=True ):
        
        self.__cuda = cuda
        self.__dictionary = {0: self.__createConv2d, 1: self.__createLinear, 2: self.__createCrossEntropyLoss}


    def findValue(self, tupleBody):
        key = tupleBody[0]

        value = self.__dictionary[key]

        return value(tupleBody)

    def __createConv2d(self, tupleBody):

        layer = torch.nn.Conv2d(tupleBody[1], tupleBody[2], tupleBody[3], tupleBody[4])
        valueLayerConv2d = torch.rand(1, tupleBody[2], 1, 1, dtype=torch.float32, requires_grad=True)
        
        self.__verifyCuda(layer)
        self.__verifyCuda(valueLayerConv2d)
        
        value = ly.Layer(objectTorch=layer, propagate=functions.conv2d_propagate, value=valueLayerConv2d)
        
        return value

    def __createLinear(self, tupleBody):
        
        layer = torch.nn.Linear(tupleBody[1], tupleBody[2])
        valueLayerLinear = torch.rand(2, dtype=torch.float32, requires_grad=True)

        self.__verifyCuda(layer)
        self.__verifyCuda(valueLayerLinear)
        
        value = ly.Layer(objectTorch=layer, propagate=functions.linear_propagate, value=valueLayerLinear)

        return value

    def __createCrossEntropyLoss(self, tupleBody):

        layer = torch.nn.CrossEntropyLoss()
        
        self.__verifyCuda(layer)

        value = ly.Layer(objectTorch=layer, propagate=functions.MSEloss_propagate)

        return value


    def __verifyCuda(self, layer):

        if self.__cuda == True:
            layer.cuda()