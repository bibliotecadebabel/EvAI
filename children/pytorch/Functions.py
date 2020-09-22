import numpy as np
import random
import torch

def Propagation(layer, n=None):

    if n is None:
        
        for parent in layer.node.parents:

            layerParent = parent.objects[0]
            Propagation(layerParent)

        layer.propagate(layer)
    else:
        if n == 0:
            pass
        
        else:
            n -= 1
            for parent in layer.node.parents:

                layerParent = parent.objects[0]
                Propagation(layerParent, n)

            layer.propagate(layer)

def BackPropagation(layer):

    for kid in layer.node.kids:
        kidLayer = kid.objects[0]
        BackPropagation(kidLayer)

    layer.backPropagate(layer)

def Nothing(layer):
    pass

############### FUNCIONES PROPAGATE ###############

def conv2d_propagate(layer):
    print("ERROR NO PROPAGATE")
    pass

def conv2d_propagate_images(layer): ## MUTATION: ADDING IMAGE TO INPUT IN EVERY CONVOLUTION LAYER
    print("ERROR NO PROPAGATE")
    pass
    
def conv2d_propagate_multipleInputs(layer): ## MUTATION: Multiple inputs per convolutional layer
    
    #parent = layer.node.parents[0].objects[0]

    current_input = __getInput(layer)

    if layer.get_pool() is not None:
        current_input = layer.apply_pooling(current_input)

    if layer.dropout_value > 0:
        output_dropout = layer.apply_dropout(current_input)
        value = layer.object(output_dropout)
    else:
        value = layer.object(current_input)

    value = layer.apply_normalization(value)
    
    layer.value = value
    
    if layer.enable_activation == True:
        layer.value = torch.nn.functional.relu(value)

def conv2d_propagate_padding(layer):
    
    #parent = layer.node.parents[0].objects[0]

    current_input = __getInput(layer)

    if layer.get_pool() is not None:
        current_input = layer.apply_pooling(current_input)

    kernel = layer.adn[3]
    #print("kernel: ", kernel)
    
    #print("original shape: ", current_input.shape)

    if layer.node.kids[0].objects[0].adn[0] == 0:
        current_input = __padInput(targetTensor=current_input, kernel_size=kernel)

    if layer.dropout_value > 0:
        output_dropout = layer.apply_dropout(current_input)
        value = layer.object(output_dropout)
    else:
        value = layer.object(current_input)

    #print("output shape: ", value.shape)
    value = layer.apply_normalization(value)
    
    layer.value = value
    
    if layer.enable_activation == True:
        layer.value = torch.nn.functional.relu(value)

def linear_propagate(layer):

    parent = layer.node.parents[0].objects[0]

    shape = parent.value.shape

    value = parent.value.view(shape[0], -1 )

    if layer.dropout_value > 0:
        output_dropout = layer.apply_dropout(value)
        value = layer.object(output_dropout)
    else:
        value = layer.object(value)

    layer.value = value

def MSEloss_propagate(layer):

    parent = layer.node.parents[0].objects[0]

    value = parent.value

    if layer.get_ricap() != None and layer.get_enable_ricap() == True:
        layer.value = layer.get_ricap().generateLoss(layer)
    else:
        layer.value = layer.object(value, layer.labels)

############### CREADOR DE TENSORES ###############

def createTensorZero(shape):

    return np.zeros((shape), dtype=float)

def createTensorRandom(shape):


    if isinstance(shape, int):
        shape = [shape]

    return np.random.rand(*shape)

def generateNumberRandom():

    return random.random()


# Crear nuevo tensor conservando valores del viejo  tensor
def createNewTensor(oldTensor, newShape):

    newFilter = createTensorRandom((newShape))

    # Conservo los valores del filtro viejo
    for i in range(len(oldTensor)):
        newFilter[i] = oldTensor[i]

    return newFilter

def deleteLastTensorDimension(oldTensor, newShape):

    newFilter = np.zeros(tuple(newShape), dtype=float)

    for i in range(newShape[0]):
        newFilter[i] = oldTensor[i]

    return newFilter

def __padInput(targetTensor, kernel_size):
    pad_tensor = targetTensor
    
    target_shape = targetTensor.shape
    refference_size = kernel_size - 1

    if refference_size > 0:
        pad_tensor = torch.nn.functional.pad(targetTensor,(0, refference_size, 0, refference_size),"constant", 0)
    
    #print("pad shape: ", pad_tensor.shape)
    return pad_tensor

def __getBiggestKernelInput(layerList):

    kernel = 0
    biggest_input = None

    for layer in layerList:
        
        shape = layer.value.shape
        
        if kernel < shape[2]:

            kernel = shape[2]
            biggest_input = layer.value
    
    return biggest_input

def __getBiggestDepthInput(layerList):

    depth = 0
    biggest_input = None

    for layer in layerList:
        
        shape = layer.value.shape
        
        if depth < shape[1]:

            depth = shape[1]
            biggest_input = layer.value
    
    return biggest_input

def __doPadKernel(targetTensor, refferenceTensor):

    pad_tensor = targetTensor
    
    target_shape = targetTensor.shape
    refference_shape = refferenceTensor.shape

    diff_kernel = abs(refference_shape[2] - target_shape[2])

    if diff_kernel > 0:
        pad_tensor = torch.nn.functional.pad(targetTensor,(0, diff_kernel, 0, diff_kernel),"constant", 0)
    
    return pad_tensor

def __doPadDepth(targetTensor, refferenceTensor):

    pad_tensor = targetTensor
    
    target_shape = targetTensor.shape
    refference_shape = refferenceTensor.shape

    diff_depth = abs(refference_shape[1] - target_shape[1])

    if diff_depth > 0:
        pad_tensor = torch.nn.functional.pad(targetTensor,(0, 0, 0, 0, 0, diff_depth),"constant", 0)
    
    return pad_tensor

def __getInput(layer):
    
    input_channels = layer.adn[1]

    parents_outputs_channels = 0

    value = None
    
    for parent_layer in layer.connected_layers:  
        parents_outputs_channels += parent_layer.value.shape[1]
    
    biggest_input_kernel = __getBiggestKernelInput(layer.connected_layers)

    if input_channels == parents_outputs_channels:

        concat_tensor_list = []

        for i in range(len(layer.connected_layers)):
            
            current_input = layer.connected_layers[i].value.clone()
            padded_input = __doPadKernel(current_input, biggest_input_kernel)
            del current_input
            concat_tensor_list.append(padded_input)

        value = torch.cat(tuple(concat_tensor_list), dim=1)

        for tensorPadded in concat_tensor_list:
            del tensorPadded
        
        del concat_tensor_list
    else:

        biggest_input_depth = __getBiggestDepthInput(layer.connected_layers) 

        sum_tensor_list = []
        #print("h: ", layer.tensor_h)

        for i in range(len(layer.connected_layers)):
            
            current_input = layer.connected_layers[i].value.clone()
            padded_input_kernel = __doPadKernel(current_input, biggest_input_kernel)
            padded_input_depth = __doPadDepth(padded_input_kernel, biggest_input_depth)
            del current_input, padded_input_kernel
            
            if i == 0:
                padded_input_depth = padded_input_depth * layer.tensor_h
            
            elif i == (len(layer.connected_layers) - 1):
                padded_input_depth = padded_input_depth * (1 - layer.tensor_h)

            sum_tensor_list.append(padded_input_depth)

        value = sum_tensor_list[0]
        for tensor in sum_tensor_list[1:]:
            value += tensor

        for tensorPadded in sum_tensor_list:
            del tensorPadded
        
        del sum_tensor_list        

    return value

