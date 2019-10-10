class Layer():

    def __init__(self, node, objectTorch=None, propagate=None, value=None, label=None):
        self.object = objectTorch
        self.node = node
        self.value =  value
        self.propagate = propagate
        self.label = label
    
    def getFilter(self):

        print("Getting Filter")        
        return self.__getParamValue(0)

    def getBias(self):

        print("Getting Bias")
        return self.__getParamValue(1)

    def __getParamValue(self, index):

        value = None

        paramList = list(self.object.parameters())
        if self.object is not None and len(paramList) > 0:
            value = paramList[index].data

        print(value)
        return value