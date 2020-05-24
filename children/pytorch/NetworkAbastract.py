from abc import ABC, abstractmethod
import Factory.LayerFactory as factory
import Factory.TensorFactory as tensorFactory

class NetworkAbstract(ABC):

    def __init__(self, adn, cuda, momentum, weight_decay, enable_activaiton, enable_track_stats=True):
        self.cudaFlag = cuda
        self.adn = adn
        self.nodes = []
        self.enable_track_stats = enable_track_stats
        self.momentum = momentum 
        self.weight_decay = weight_decay
        self.enable_activation = enable_activaiton 
        self.label = tensorFactory.createTensor(body=[1], cuda=self.cudaFlag, requiresGrad=False)
        self.factory = factory.LayerGenerator(cuda=self.cudaFlag)
        self.foward_value = None   
        self.total_value = 0
        self.history_loss = []

    #@abstractmethod
    #def train(self):
    #    pass

    def setAttribute(self, name, value):
        setattr(self,name,value)

    def getAttribute(self, name):

        attribute = None
        try:
            attribute = getattr(self, name)
        except AttributeError:
            pass

        return attribute

    def deleteAttribute(self, name):
        try:
            delattr(self, name)
        except AttributeError:
            pass

    def updateGradFlag(self, flag):

        for node in self.nodes[:-1]:

            layer = node.objects[0]

            if layer.getBias() is not None:
                layer.getBias().requires_grad = flag

            if layer.getBiasDer() is not None:
                layer.getBiasDer().requires_grad = flag

            if layer.getFilter() is not None:
                layer.getFilter().requires_grad = flag

            if layer.getFilterDer() is not None:
                layer.getFilterDer().requires_grad = flag
    
    def assignLabels(self, label):

        for node in self.nodes:
            node.objects[0].label = label

    def getAverageLoss_test(self, iterations):

        average = self.__accumulated_loss / iterations

        self.__accumulated_loss = 0

        return average
    
    def getAverageLoss(self, iterations):

        last_x = self.history_loss[-iterations:]

        accum = 0

        for loss in last_x:
            accum += loss
        
        return (accum/iterations)
    
    def getLossArray(self):

        value = self.history_loss.copy()
        self.history_loss = []
        return value