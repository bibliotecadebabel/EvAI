'''
import Geometric.Directions.DNA_directions_convex as dna_directions

dna = ((-1, 1, 3, 32, 32), (0, 3, 4, 3, 3), (0, 4, 5, 3, 3, 2), (0, 5, 5, 3, 3), (0, 5, 6, 3, 3, 2), (0, 6, 7, 3, 3), (0, 7, 8, 8, 8), (1, 8, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 6, 7))

new_dna = dna_directions.add_layer(3,dna)

print("new dna: ", new_dna)
'''

from children.pytorch.layers.loss_layers import layer_loss
from  children.pytorch.layers.learnable_layers import layer_conv2d, layer_learnable, layer_linear

conv2d = layer_conv2d.Conv2dLayer(adn=(1,1,1), node=None, torch_object=None, enable_activation=True, propagate_mode=1)
linear = layer_linear.LinearLayer(adn=(2,2,2), node=None, torch_object=None)
loss = layer_loss.LossLayer(adn=(3,3,3), node=None, torch_object=None)


if isinstance(conv2d, layer_learnable.LearnableLayer):
    print("conv2d is learnable")

if isinstance(linear, layer_learnable.LearnableLayer):
    print("linear is learnable")

if isinstance(loss, layer_learnable.LearnableLayer) == False:
    print("loss is NOT learnable")

