class Layer():

    def __init__(self, node, objectTorch=None, propagate=None, value=None, label=None):
        self.object = objectTorch
        self.node = node
        self.value =  value
        self.propagate = propagate
        self.label = label

        self.bias_der_total = 0
        self.filter_der_total = 0

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

    def getBias(self):

        value = None
        
        if self.object is not None:
            value = self.object.bias

        return value

    def __getParamValue(self, index, grad):

        value = None

        paramList = list(self.object.parameters())

        if self.object is not None and len(paramList) > 0:
            if grad == False:
                value = paramList[index].data
            else:
                value = paramList[index].grad.data

        return value