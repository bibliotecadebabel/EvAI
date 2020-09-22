from abc import ABC, abstractmethod
import Factory.LayerFactory as factory
import Factory.TensorFactory as tensorFactory
import children.pytorch.layers.learnable_layers.layer_learnable as layer_learnable

class NetworkAbstract(ABC):

    def __init__(self, adn, cuda, momentum, weight_decay, enable_activaiton, enable_track_stats=True, dropout_value=0, enable_last_activation=True):
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
        self.dropout_value = dropout_value
        self.enable_last_activation = enable_last_activation

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

            if isinstance(layer, layer_learnable.LearnableLayer):

                if layer.get_bias() is not None:
                    layer.get_bias().requires_grad = flag

                if layer.get_bias_grad() is not None:
                    layer.get_bias_grad().requires_grad = flag

                if layer.get_filters() is not None:
                    layer.get_filters().requires_grad = flag

                if layer.get_filters_grad() is not None:
                    layer.get_filters_grad().requires_grad = flag
                
                if layer.tensor_h is not None:
                    layer.tensor_h.requires_grad = flag
                
    
    def assignLabels(self, labels):

        self.nodes[len(self.nodes)-1].objects[0].set_labels(labels)
    
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
    
    def getLossLayer(self):

        return self.nodes[len(self.nodes)-1].objects[0]

    def getLayerProbability(self):

        return self.nodes[len(self.nodes)-2].objects[0]