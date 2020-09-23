import children.pytorch.layers.layer as layer
from abc import abstractmethod
import torch

class LearnableLayer(layer.Layer):

    def __init__(self, dna, torch_object):
        
        layer.Layer.__init__(self, dna=dna)
        
        self.object = torch_object
        self.__dropout_value = 0
        self.__dropout = None
        self.tensor_h = None
    
    def get_dropout_value(self):
        return self.__dropout_value

    def set_dropout_value(self, value):
        self.__dropout_value = value
    
    def set_dropout(self, value):
        self.__dropout = value
    
    def get_dropout(self):
        return self.__dropout

    def get_bias_grad(self):

        value = None

        if self.dna is not None:
            if self.dna[0] == 0 or self.dna[0] == 1:
                value = self.object.bias.grad

        return value

    def get_filters_grad(self):

        value = None
        
        if self.dna is not None:
            if self.dna[0] == 0 or self.dna[0] == 1:
                value = self.object.weight.grad

        return value
       

    def get_filters(self):
 
        value = None
        
        if self.dna is not None:
            if self.dna[0] == 0 or self.dna[0] == 1:
                value = self.object.weight

        return value
    
    def set_filters(self, value):

        if self.dna is not None:
            if self.dna[0] == 0 or self.dna[0] == 1:
                self.object.weight = torch.nn.Parameter(value)

    def get_bias(self):

        value = None
        
        if self.dna is not None:
            if self.dna[0] == 0 or self.dna[0] == 1:
                value = self.object.bias

        return value
    
    def set_bias(self, value):

        if self.dna is not None:
            if self.dna[0] == 0 or self.dna[0] == 1:
                self.object.bias = torch.nn.Parameter(value)

    def apply_dropout(self, tensor):
        
        dropout = self.__dropout(tensor)
        return dropout

    @layer.abstractmethod
    def propagate(self):
        pass
    
    @layer.abstractmethod
    def deleteParam(self):
        pass