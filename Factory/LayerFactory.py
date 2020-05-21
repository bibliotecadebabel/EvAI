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
        batchNormalization = torch.nn.BatchNorm2d(tupleBody[2])

        self.__verifyCuda(batchNormalization)
        self.__verifyCuda(layer)
        
        if propagate_mode == const.CONV2D_MULTIPLE_INPUTS:
            value = ly.Layer(objectTorch=layer, propagate=functions.conv2d_propagate_multipleInputs, value=None, adn=tupleBody, 
                cudaFlag=self.__cuda, batchNorm=batchNormalization, enable_activation=enable_activation)
        elif propagate_mode == const.CONV2D_IMAGE_INPUTS:
            value = ly.Layer(objectTorch=layer, propagate=functions.conv2d_propagate_images, value=None, adn=tupleBody, 
                cudaFlag=self.__cuda, batchNorm=batchNormalization, enable_activation=enable_activation)
        else:
            value = ly.Layer(objectTorch=layer, propagate=functions.conv2d_propagate, value=None, adn=tupleBody, 
                cudaFlag=self.__cuda, batchNorm=batchNormalization, enable_activation=enable_activation)

        return value

    def __createLinear(self, tupleBody, propagate_mode=None, enable_activation=False):
        
        layer = torch.nn.Linear(tupleBody[1], tupleBody[2])
        self.__initLinear(layer, tupleBody[1])

        self.__verifyCuda(layer)
        
        value = ly.Layer(objectTorch=layer, adn=tupleBody,propagate=functions.linear_propagate, value=None, cudaFlag=self.__cuda)

        return value

    def __createCrossEntropyLoss(self, tupleBody, propagate_mode=None, enable_activation=False):

        layer = torch.nn.CrossEntropyLoss()
        
        self.__verifyCuda(layer)

        value = ly.Layer(objectTorch=layer, adn=tupleBody, propagate=functions.MSEloss_propagate, cudaFlag=self.__cuda)

        return value


    def __verifyCuda(self, layer):

        if self.__cuda == True:
            layer.cuda()

    def __initConv2d(self, layer, kernel_shape):

        #kernel_product = math.sqrt(kernel_shape[0] * kernel_shape[1] * kernel_shape[2])

        #torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
        #torch.nn.init.xavier_uniform_(layer.bias)
        '''
        torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)
        layer.weight = torch.nn.Parameter(torch.div(torch.mul(layer.weight, math.sqrt(2)), kernel_product).clone())
        layer.bias = torch.nn.Parameter(torch.div(torch.mul(layer.bias, math.sqrt(2)), kernel_product).clone())
            '''

    def __initLinear(self, layer, entrys):

        #entry_product = math.sqrt(entrys)
    
  
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
        #torch.nn.init.xavier_uniform_(layer.bias)
        '''
        torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)
        layer.weight = torch.nn.Parameter(torch.div(torch.mul(layer.weight, math.sqrt(2)), entry_product).clone())
        layer.bias = torch.nn.Parameter(torch.div(torch.mul(layer.bias, math.sqrt(2)), entry_product).clone())
        '''
