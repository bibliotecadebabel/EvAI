from abc import ABC, abstractmethod

class Layer(ABC):

    def __init__(self, adn=None, node=None, torch_object=None):
        self.object = torch_object
        self.node = node
        self.value = None
        self.connected_layers = []
        self.adn = adn
        
        #self.__batchnorm = None # CONV2D
        #self.__pool = None # CONV2D
    
    @abstractmethod
    def propagate(self):
        pass

    @abstractmethod
    def deleteParam(self):
        pass