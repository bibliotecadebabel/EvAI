import torch
import children.pytorch.Layer as ly
import children.pytorch.Functions as functions
import Factory.AbstractFactory as AbstractFactory
import const.propagate_mode as const
import math

class LayerGenerator(AbstractFactory.FactoryClass):

    def __init__( self, cuda=True ):
        super().__init__()
        
        self.__cuda = cuda
        self.__track_stats = None

    def createDictionary(self):

        self.dictionary[0] = self.__createConv2d
        self.dictionary[1] = self.__createLinear
        self.dictionary[2] = self.__createCrossEntropyLoss

    def findValue(self, tupleBody, propagate_mode, enable_activation):
        key = tupleBody[0]

        value = self.dictionary[key]

        return value(tupleBody, propagate_mode, enable_activation)

    def __createConv2d(self, tupleBody, propagate_mode, enable_activation):
        
        layer = torch.nn.Conv2d(tupleBody[1], tupleBody[2], (tupleBody[3], tupleBody[4]))
        self.__initConv2d(layer, (1, tupleBody[3], tupleBody[4]))

        self.__verifyCuda(layer)
        
        if propagate_mode == const.CONV2D_PADDING:
            value = ly.Layer(torch_object=layer, propagate=functions.conv2d_propagate_padding, adn=tupleBody, 
                                enable_activation=enable_activation)
        else:
            value = ly.Layer(torch_object=layer, propagate=functions.conv2d_propagate_multipleInputs, adn=tupleBody,
                                enable_activation=enable_activation)

        return value

    def __createLinear(self, tupleBody, propagate_mode=None, enable_activation=False):
        
        layer = torch.nn.Linear(tupleBody[1], tupleBody[2])
        self.__initLinear(layer, tupleBody[1])

        self.__verifyCuda(layer)
        
        value = ly.Layer(torch_object=layer, adn=tupleBody,propagate=functions.linear_propagate)

        return value

    def __createCrossEntropyLoss(self, tupleBody, propagate_mode=None, enable_activation=False):

        layer = torch.nn.CrossEntropyLoss()
        
        self.__verifyCuda(layer)

        value = ly.Layer(torch_object=layer, adn=tupleBody, propagate=functions.MSEloss_propagate)

        return value

    def __verifyCuda(self, layer):

        if self.__cuda == True:
            layer.cuda()

    def __initConv2d(self, layer, kernel_shape):
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

    def __initLinear(self, layer, entrys):
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
