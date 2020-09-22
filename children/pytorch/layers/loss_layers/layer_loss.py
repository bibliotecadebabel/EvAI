import children.pytorch.layers.layer as layer

class LossLayer(layer.Layer):

    def __init__(self, adn, torch_object):
        
        layer.Layer.__init__(self, adn=adn)
        
        self.object = torch_object
        self.__labels = None
        self.__crops = None
        self.__ricap = None
        self.__enable_ricap = False

    def set_labels(self, labels):
        self.__labels = labels

    def set_crops(self, value):
        self.__crops = value

    def get_crops(self):
        return self.__crops

    def get_enable_ricap(self):
        return self.__enable_ricap
    
    def set_enable_ricap(self, value):
        self.__enable_ricap = value

    def set_ricap(self, value):
        self.__ricap = value
        
    def get_ricap(self):
        return self.__ricap
        
    def propagate(self):
        
        parent = self.node.parents[0].objects[0]

        value = parent.value

        if self.get_ricap() != None and self.get_enable_ricap() == True:
            self.value = self.get_ricap().generateLoss(self)
        else:
            self.value = self.object(value, self.__labels)

    def deleteParam(self):

        if self.__labels is not None:
            del self.__labels