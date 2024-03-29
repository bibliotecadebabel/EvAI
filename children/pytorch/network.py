from abc import ABC, abstractmethod
import Factory.TensorFactory as tensorFactory
import children.pytorch.layers.learnable_layers.layer_learnable as layer_learnable

class NetworkAbstract(ABC):

    def __init__(self, dna, cuda, momentum, weight_decay, enable_activaiton, enable_track_stats=True, dropout_value=0, enable_last_activation=True):
        self.cuda_flag = cuda
        self.dna = dna
        self.nodes = []
        self.enable_track_stats = enable_track_stats
        self.momentum = momentum 
        self.weight_decay = weight_decay
        self.enable_activation = enable_activaiton 
        self.label = tensorFactory.createTensor(body=[1], cuda=self.cuda_flag, requiresGrad=False)
        self.foward_value = None   
        self.total_value = 0
        self.loss_history = []
        self.dropout_value = dropout_value
        self.enable_last_activation = enable_last_activation

    def set_attribute(self, name, value):
        setattr(self,name,value)

    def set_grad_flag(self, flag):

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
                
    
    def assign_labels(self, labels):

        self.nodes[len(self.nodes)-1].objects[0].set_labels(labels)
    
    def get_average_loss(self, iterations):

        last_x = self.loss_history[-iterations:]

        accum = 0

        for loss in last_x:
            accum += loss
        
        return (accum/iterations)
    
    def get_loss_array(self):

        value = self.loss_history.copy()
        self.loss_history = []
        return value
    
    def get_loss_layer(self):

        return self.nodes[len(self.nodes)-1].objects[0]

    def get_linear_layer(self):

        return self.nodes[len(self.nodes)-2].objects[0]