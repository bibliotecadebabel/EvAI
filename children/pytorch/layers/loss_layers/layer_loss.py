import children.pytorch.layers.layer_torch as layer

class LossLayer(layer.TorchLayer):

    def __init__(self, dna, torch_object):
        
        layer.TorchLayer.__init__(self, dna=dna, torch_object=torch_object)
        
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

        # Se calcula el valor de pérdida, dependiendo si aplica o no RICAP.
        if self.get_ricap() != None and self.get_enable_ricap() == True:
            self.value = self.get_ricap().generateLoss(self)
        else:
            self.value = self.object(value, self.__labels)
        
    def delete_params(self):

        if self.__labels is not None:
            del self.__labels