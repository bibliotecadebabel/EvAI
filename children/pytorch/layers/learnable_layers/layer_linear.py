import children.pytorch.layers.learnable_layers.layer_learnable as layer
import const.propagate_mode
from abc import abstractmethod
import torch

class LinearLayer(layer.LearnableLayer):

    def __init__(self, dna, torch_object):
        
        layer.LearnableLayer.__init__(self, dna=dna, torch_object=torch_object)

    # Función de propagación Linear
    def propagate(self):

        parent = self.node.parents[0].objects[0]
        
        shape = parent.value.shape

        value = parent.value.view(shape[0], -1 )

        if self.get_dropout_value() > 0:
            # Se aplica Dropout y luego la función lineal
            output_dropout = self.apply_dropout(value)
            value = self.object(output_dropout)
        else:
            # Se aplica la función lineal
            value = self.object(value)

        self.value = value
    
    def delete_params(self):

        if self.object is not None:

            if hasattr(self.object, 'weight') and self.object.weight is not None:

                if hasattr(self.object.weight, 'grad') and self.object.weight.grad is not None:
                    del self.object.weight.grad

                del self.object.weight
            
            if hasattr(self.object, 'bias') and self.object.bias is not None:

                if hasattr(self.object.bias, 'grad') and self.object.bias.grad is not None:
                    del self.object.bias.grad

                del self.object.bias
            
            del self.object