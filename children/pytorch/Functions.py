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

    parent = layer.node.parents[0].objects[0]
    
    value = layer.object(parent.value)

    value = layer.doNormalize(value)
    
    layer.value = value
    
    if layer.enable_activation == True:
        
        sigmoid = torch.nn.Sigmoid()
        layer.value = sigmoid(value) + torch.nn.functional.relu(value)
    
def conv2d_propagate_images(layer): ## MUTATION: ADDING IMAGE TO INPUT IN EVERY CONVOLUTION LAYER
    
    parent = layer.node.parents[0].objects[0]

    if parent.adn is None:
        layer.image = parent.value
    else:
        layer.image = parent.image

    kid = layer.node.kids[0].objects[0]  

    value = layer.object(parent.value)
    
    value = layer.doNormalize(value)

    layer.value = value
    
    if layer.enable_activation == True:
        
        sigmoid = torch.nn.Sigmoid()
        layer.value = sigmoid(value) + torch.nn.functional.relu(value)

    if kid.adn is not None and kid.adn[0] == 0: #Check if is conv2d

        inputShape = layer.image.shape
        outputShape = layer.value.shape
        
        diff_kernel = abs(inputShape[2] - outputShape[2])
        
        if inputShape[2] >= outputShape[2]:
            
            with torch.no_grad():
                newValue = layer.value.data.clone()
                newValue = torch.nn.functional.pad(newValue,(0, diff_kernel, 0, diff_kernel),"constant", 0)

                layer.value = torch.cat((layer.image, newValue), dim=1)
        else:
            print("OUTPUT LARGER THAN INPUTS")
        
def conv2d_propagate_multipleInputs(layer): ## MUTATION: Multiple inputs per convolutional layer
    
    parent = layer.node.parents[0].objects[0]
    
    #print("current layer= ", layer.adn)
    current_input = __getInput(layer, parent.value)

    value = layer.object(current_input) 
    value = layer.doNormalize(value)
    
    layer.value = value
    
    if layer.enable_activation == True:
        
        sigmoid = torch.nn.Sigmoid()
        layer.value = sigmoid(value) + torch.nn.functional.relu(value)
    

def linear_propagate(layer):

    parent = layer.node.parents[0].objects[0]

    shape = parent.value.shape
    
    #print("value last conv2d= ", shape)

    layer.value = layer.object(parent.value.view(shape[0], -1 ))

def MSEloss_propagate(layer):

    parent = layer.node.parents[0].objects[0]

    layer.value = layer.object(parent.value, layer.label)

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

def __getBiggestInput(layerList):

    kernel = 0
    biggest_input = None

    for layer in layerList:
        
        shape = layer.value.shape
        
        if kernel < shape[2]:

            kernel = shape[2]
            biggest_input = layer.value
    
    #if kernel < currentInput.shape[2]:
    #    biggest_input = currentInput
    
    return biggest_input

def __doPad(targetTensor, refferenceTensor):

    pad_tensor = targetTensor
    
    target_shape = targetTensor.shape
    refference_shape = refferenceTensor.shape

    diff_kernel = abs(refference_shape[2] - target_shape[2])

    if diff_kernel > 0:
        pad_tensor = torch.nn.functional.pad(targetTensor,(0, diff_kernel, 0, diff_kernel),"constant", 0)
        del targetTensor
    
    return pad_tensor

def __getInput(layer, parentOutput):
    
    len_other_inputs = len(layer.other_inputs)

    value = parentOutput

    if len_other_inputs > 0:
        
        with torch.no_grad():   

            #normal_input = parentOutput.clone()

            #biggest_input = __getBiggestInput(normal_input, layer.other_inputs)
            
            biggest_input = __getBiggestInput(layer.other_inputs)
            #normal_input = __doPad(normal_input, biggest_input)

            concat_tensor_list = []

            for i in range(len(layer.other_inputs)):
                
                #print("concat layer=", layer.other_inputs[i].adn)
                current_input = layer.other_inputs[i].value.clone()
                #print("concat input= ", current_input.shape)

                current_input = __doPad(current_input, biggest_input)
                #print("concat input padded= ", current_input.shape)

                concat_tensor_list.append(current_input)


            value = torch.cat(tuple(concat_tensor_list), dim=1)

            #print("final input size=", value.shape)

            for tensorPadded in concat_tensor_list:
                del tensorPadded
            
            del concat_tensor_list

    return value

