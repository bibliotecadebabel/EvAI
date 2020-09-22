import torch
from children.pytorch.layers.layer import Layer
from children.pytorch.layers.loss_layers import layer_loss
from children.pytorch.layers.learnable_layers import layer_conv2d, layer_linear
import children.pytorch.Functions as functions
import Factory.AbstractFactory as AbstractFactory
import const.propagate_mode as const
import math

class LayerGenerator(AbstractFactory.FactoryClass):

    def __init__( self, cuda=True):
        super().__init__()
        
        self.__cuda = cuda
        self.__enable_activation = False
        self.__propagate_mode = False
        self.__layer_tuple = None

    def createDictionary(self):

        self.dictionary[-1] = self.__createImage
        self.dictionary[0] = self.__createConv2d
        self.dictionary[1] = self.__createLinear
        self.dictionary[2] = self.__createCrossEntropyLoss

    def findValue(self, layer_tuple, propagate_mode, enable_activation):

        key = layer_tuple[0]

        value = self.dictionary[key]

        self.__propagate_mode = propagate_mode
        self.__enable_activation = enable_activation
        self.__layer_tuple = layer_tuple

        layer = value()

        return layer

    def __createImage(self):

        layer = Layer(adn=self.__layer_tuple)

        return layer

    def __createConv2d(self):
        
        torch_object = torch.nn.Conv2d(self.__layer_tuple[1], self.__layer_tuple[2], (self.__layer_tuple[3], self.__layer_tuple[4]))

        self.__initConv2d(torch_object)
        self.__verifyCuda(torch_object)
        
        layer = layer_conv2d.Conv2dLayer(adn=self.__layer_tuple, torch_object=torch_object, enable_activation=self.__enable_activation,
                                            propagate_mode=self.__propagate_mode)

        return layer

    def __createLinear(self):
        
        torch_object = torch.nn.Linear(self.__layer_tuple[1], self.__layer_tuple[2])

        self.__initLinear(torch_object)
        self.__verifyCuda(torch_object)
        
        value = layer_linear.LinearLayer(adn=self.__layer_tuple, torch_object=torch_object)

        return value

    def __createCrossEntropyLoss(self):

        torch_object = torch.nn.CrossEntropyLoss()
        
        self.__verifyCuda(torch_object)

        value = layer_loss.LossLayer(adn=self.__layer_tuple, torch_object=torch_object)

        return value

    def __verifyCuda(self, layer):

        if self.__cuda == True:
            layer.cuda()

    def __initConv2d(self, torch_object):

        torch.nn.init.xavier_uniform_(torch_object.weight)
        torch.nn.init.zeros_(torch_object.bias)

    def __initLinear(self, torch_object):
        
        torch.nn.init.xavier_uniform_(torch_object.weight)
        torch.nn.init.zeros_(torch_object.bias)
