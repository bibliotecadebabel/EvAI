import torch

class Layer():

    def __init__(self, adn=None, node=None, objectTorch=None, propagate=None, value=None, label=None, cudaFlag=True):
        self.object = objectTorch
        self.node = node
        self.value =  value
        self.propagate = propagate
        self.label = label
        self.cudaFlag = cudaFlag
        self.image = None

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
        
    def getBiasDer(self):

        value = None

        if self.object is not None:
            value = self.object.bias.grad

        return value

    def getFilterDer(self):

        value = None
        
        if self.object is not None:
            value = self.object.weight.grad

        return value
       

    def getFilter(self):
 
        value = None
        
        if self.object is not None:
            value = self.object.weight

        return value
    
    def setFilter(self, value):

        if self.object is not None:
            self.object.weight = torch.nn.Parameter(value)

    def getBias(self):

        value = None
        
        if self.object is not None:
            value = self.object.bias

        return value
    
    def setBias(self, value):

        if self.object is not None:
            self.object.bias = torch.nn.Parameter(value)

    def __getParamValue(self, index, grad):

        value = None

        paramList = list(self.object.parameters())

        if self.object is not None and len(paramList) > 0:
            if grad == False:
                value = paramList[index].data
            else:
                value = paramList[index].grad.data

        return value