import torch

class Layer():

    def __init__(self, adn=None, node=None, torch_object=None, propagate=None, value=None, label=None, cudaFlag=True, enable_activation=True, dropout_value=0):
        self.object = torch_object
        self.node = node
        self.value =  value
        self.propagate = propagate
        self.label = label
        self.cudaFlag = cudaFlag
        self.image = None
        self.other_inputs = []
        self.dropout_value = dropout_value
        self.tensor_h = None
            
        self.bias_der_total = 0
        self.filter_der_total = 0
        self.adn = adn
        self.enable_activation = enable_activation
        self.__crops = 0
        
        self.__batchnorm = None
        self.__dropout = None
        self.__pool = None
        self.__ricap = None
        self.__enableRicap = False

    def get_enable_ricap(self):
        return self.__enableRicap
    
    def set_enable_ricap(self, value):
        self.__enableRicap = value

    def set_ricap(self, value):
        self.__ricap = value
        
    def get_ricap(self):
        return self.__ricap

    def set_crops(self, value):
        self.__crops = value

    def get_crops(self):
        return self.__crops

    def get_bias_grad(self):

        value = None

        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                value = self.object.bias.grad

        return value

    def get_filters_grad(self):

        value = None
        
        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                value = self.object.weight.grad

        return value
       

    def get_filters(self):
 
        value = None
        
        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                value = self.object.weight

        return value
    
    def set_filters(self, value):

        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                self.object.weight = torch.nn.Parameter(value)

    def get_bias(self):

        value = None
        
        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                value = self.object.bias

        return value
    
    def set_bias(self, value):

        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                self.object.bias = torch.nn.Parameter(value)
    
    def set_pool(self, object_torch):
        self.__pool = object_torch
    
    def get_pool(self):
        return self.__pool

    def set_batch_norm_object(self, object_torch):
        self.__batchnorm = object_torch
    
    def set_dropout(self, object_torch):
        self.__dropout = object_torch
    
    def get_batch_norm_object(self):
        return self.__batchnorm
    
    def get_dropout(self):
        return self.__dropout

    def set_batch_norm(self, value):

        if self.__batchnorm is not None and value is not None:
            self.__set_norm_bias(value.bias)
            self.__set_norm_weight(value.weight)
            self.__set_norm_var(value.running_var)
            self.__set_norm_mean(value.running_mean)
            self.__set_norm_batches_tracked(value.num_batches_tracked)
    
    def __set_norm_bias(self, value):
        self.__batchnorm.bias = torch.nn.Parameter(value.clone())
    
    def __set_norm_weight(self, value):
        self.__batchnorm.weight = torch.nn.Parameter(value.clone())
    
    def __set_norm_var(self, value):

        if value is not None:
            self.__batchnorm.running_var.data = value.data.clone()

    def __set_norm_mean(self, value):

        if value is not None:
            self.__batchnorm.running_mean.data = value.data.clone()
    
    def __set_norm_batches_tracked(self, value):

        if value is not None:
            self.__batchnorm.num_batches_tracked.data = value.data.clone()

    def get_batch_norm(self):

        return self.__batchnorm

    def apply_normalization(self, tensor):

        norm = self.__batchnorm(tensor)
        return norm
    
    def apply_pooling(self, tensor):
        
        value = self.__pool(tensor)
        return value

    def apply_dropout(self, tensor):
        
        dropout = self.__dropout(tensor)
        return dropout
    
    def deleteParam(self):

        if self.label is not None:
            del self.label

        if self.object is not None:

            if hasattr(self.object, 'weight') and self.object.weight is not None:

                if hasattr(self.object.weight, 'grad') and self.object.weight.grad is not None:
                    del self.object.weight.grad

                del self.object.weight
            
            if hasattr(self.object, 'bias') and self.object.bias is not None:

                if hasattr(self.object.bias, 'grad') and self.object.bias.grad is not None:
                    del self.object.bias.grad

                del self.object.bias
            
            if self.__pool is not None:
                del self.__pool
            
            del self.object

            if self.__batchnorm is not None:

                if hasattr(self.__batchnorm, 'weight') and self.__batchnorm.weight is not None:
                    del self.__batchnorm.weight

                if  hasattr(self.__batchnorm, 'bias') and self.__batchnorm.bias is not None:
                    del self.__batchnorm.bias
                
                if hasattr(self.__batchnorm, 'running_var') and self.__batchnorm.running_var is not None:
                    del self.__batchnorm.running_var

                if hasattr(self.__batchnorm, 'running_mean') and self.__batchnorm.running_mean is not None:
                    del self.__batchnorm.running_mean
                
                if hasattr(self.__batchnorm, 'num_batches_tracked') and self.__batchnorm.num_batches_tracked is not None:
                    del self.__batchnorm.num_batches_tracked
            
