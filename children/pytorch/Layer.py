import torch

class Layer():

    def __init__(self, adn=None, node=None, objectTorch=None, propagate=None, value=None, label=None, cudaFlag=True, enable_activation=True, dropout_value=0):
        self.object = objectTorch
        self.node = node
        self.value =  value
        self.propagate = propagate
        self.label = label
        self.cudaFlag = cudaFlag
        self.image = None
        self.other_inputs = []
        self.dropout_value = dropout_value

        if self.cudaFlag == True:
            self.swap = torch.tensor([[0, 1], [1,0]], dtype=torch.float32, requires_grad=True).cuda()
        else:
            self.swap = torch.tensor([[0, 1], [1,0]], dtype=torch.float32, requires_grad=True)

        if self.cudaFlag == True:
            self.labelCircle = torch.tensor([0], dtype=torch.long).cuda()
        else:
            self.labelCircle = torch.tensor([0], dtype=torch.long)
            
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

    def getEnableRicap(self):
        return self.__enableRicap
    
    def setEnableRicap(self, value):
        self.__enableRicap = value

    def setRicap(self, value):
        self.__ricap = value
        
    def getRicap(self):
        return self.__ricap

    def setCrops(self, value):
        self.__crops = value

    def getCrops(self):
        return self.__crops

    def getBiasDer(self):

        value = None

        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                value = self.object.bias.grad

        return value

    def getFilterDer(self):

        value = None
        
        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                value = self.object.weight.grad

        return value
       

    def getFilter(self):
 
        value = None
        
        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                value = self.object.weight

        return value
    
    def setFilter(self, value):

        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                self.object.weight = torch.nn.Parameter(value)

    def getBias(self):

        value = None
        
        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                value = self.object.bias

        return value
    
    def setBias(self, value):

        if self.adn is not None:
            if self.adn[0] == 0 or self.adn[0] == 1:
                self.object.bias = torch.nn.Parameter(value)
    
    def setPool(self, object_torch):
        self.__pool = object_torch
    
    def getPool(self):
        return self.__pool

    def setBatchNormObject(self, object_torch):
        self.__batchnorm = object_torch
    
    def setDropoutObject(self, object_torch):
        self.__dropout = object_torch
    
    def getBatchNormObject(self):
        return self.__batchnorm
    
    def getDropoutObject(self):
        return self.__dropout

    def setBarchNorm(self, value):

        if self.__batchnorm is not None and value is not None:
            self.setBiasNorm(value.bias)
            self.setWeightNorm(value.weight)
            self.setVarNorm(value.running_var)
            self.setMeanNorm(value.running_mean)
            self.setBatchesTrackedNorm(value.num_batches_tracked)
    
    def setBiasNorm(self, value):
        self.__batchnorm.bias = torch.nn.Parameter(value.clone())
    
    def setWeightNorm(self, value):
        self.__batchnorm.weight = torch.nn.Parameter(value.clone())
    
    def setVarNorm(self, value):

        if value is not None:
            self.__batchnorm.running_var.data = value.data.clone()

    def setMeanNorm(self, value):

        if value is not None:
            self.__batchnorm.running_mean.data = value.data.clone()
    
    def setBatchesTrackedNorm(self, value):

        if value is not None:
            self.__batchnorm.num_batches_tracked.data = value.data.clone()

    def getBatchNorm(self):

        return self.__batchnorm

    def getBiasNorm(self):
        
        return self.__batchnorm.bias
    
    def getWeightNorm(self):
        
        return self.__batchnorm.weight
    
    def getVarNorm(self):
        
        return self.__batchnorm.running_var

    def getMeanNorm(self):
        
        return self.__batchnorm.running_mean

    def doNormalize(self, tensor):

        norm = self.__batchnorm(tensor)
        return norm
    
    def doPool(self, tensor):
        
        value = self.__pool(tensor)
        return value

    def doDropout(self, tensor):
        
        dropout = self.__dropout(tensor)
        return dropout

    def __getParamValue(self, index, grad):

        value = None

        paramList = list(self.object.parameters())

        if self.object is not None and len(paramList) > 0:
            if grad == False:
                value = paramList[index].data
            else:
                value = paramList[index].grad.data

        return value
    
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
            
