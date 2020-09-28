from abc import abstractmethod
import children.pytorch.layers.layer as layer

class TorchLayer(layer.Layer):

    def __init__(self, dna, torch_object):
        
        layer.Layer.__init__(self, dna=dna)

        self.object = torch_object
    
    @abstractmethod
    def propagate(self):
        pass
    
    @abstractmethod
    def delete_params(self):
        pass