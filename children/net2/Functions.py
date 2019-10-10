import numpy as np
import random

def Propagation(layer):

    for parent in layer.node.parents:
        layerParent = parent.objects[0]
        Propagation(layerParent)

    layer.propagate(layer)

def BackPropagation(layer):

    for kid in layer.node.kids:
        kidLayer = kid.objects[0]
        BackPropagation(kidLayer)

    layer.backPropagate(layer)

def Nothing(layer):
    pass

############### FUNCIONES PROPAGATE ###############

def ProductoPunto(layer):
    parent = layer.node.parents[0]

    for i in range(len(parent.objects[0].filters)):
        layer.value[i] = Dot(parent.objects[0].filters[i], (parent.objects[0].value + parent.objects[0].bias))

    for i in range(len(parent.objects[0].filters)):
        layer.bias[i] = Dot(parent.objects[0].filters[i], (parent.objects[0].value + parent.objects[0].bias))

def probability(layer):
    parent = layer.node.parents[0]
    sc =  parent.objects[0].value[0]
    sn = parent.objects[0].value[1]
    label=parent.objects[0].label
    #print(label)
    if label is None or label == "c":
        layer.value = np.exp(sc)/(np.exp(sc) + np.exp(sn))
    elif label == "n":
        layer.value = (np.exp(sn)/(np.exp(sc) + np.exp(sn)))

def logaritmo(layer):
    parent = layer.node.parents[0]
    layer.value = np.log(parent.objects[0].value)*-1

def Dot(f, v):

    y = f * v
    y = y.sum()
    y = y / (len(f)*255)

    return y

############### FUNCIONES BACKPROPAGATE ###############

def probability_der(layer):
    layer.value_der = (1/layer.value)*-1

def c_filter_der(layer):
    kid = layer.node.kids[0]
    p = kid.objects[0].value
    sc = layer.value[0]
    sn = layer.value[1]
    label=layer.label
    filter_der = np.zeros(2, dtype=float)
    if label == "c":
        filter_der[0] = p - (p*p)
        filter_der[1] = (-(p*p)*np.exp(sn))/np.exp(sc)
    else:
        filter_der[0] = (-(p*p)*np.exp(sc))/np.exp(sn)
        filter_der[1] = p - (p*p)

    layer.value_der = filter_der*kid.objects[0].value_der

def b_filter_der(layer):
    kid = layer.node.kids[0]

    layer.filter_der[0] = (layer.value+layer.bias)*kid.objects[0].value_der[0]
    layer.filter_der[1] = (layer.value+layer.bias)*kid.objects[0].value_der[1]

    layer.value_der = (((layer.filters[0] * kid.objects[0].value_der[0]))
        + ((layer.filters[1] * kid.objects[0].value_der[1])))
    layer.bias_der = (((layer.filters[0] * kid.objects[0].value_der[0]))
        + ((layer.filters[1] * kid.objects[0].value_der[1])))

def a_filter_der(layer):

    kid = layer.node.kids[0]

    for i in range(layer.filters.shape[0]):
        layer.filter_der[i] = ((layer.value+layer.bias) * kid.objects[0].value_der[i])
    layer.bias_der=0
    for i in range(layer.filters.shape[0]):
        layer.bias_der += (layer.filters[i]
            * kid.objects[0].value_der[i])

############### ELIMINAR FILTROS ###############

def removeFilters(layer):
    removeFilterNodeA(layer)
    removeFilterNodeB(layer.node.kids[0].objects[0])

def removeFilterNodeA(layerNodeA):
    if layerNodeA.filters is not None and len(layerNodeA.filters) > 0:

        newFilterShape = list(layerNodeA.filters.shape)

        if newFilterShape[0] > 0:
            newFilterShape[0] -= 1

        #layerNodeA.filters = deleteLastFilter(layerNodeA.filters, newFilterShape)

def removeFilterNodeB(layerNodeB):
    if layerNodeB.filters[0] is not None and len(layerNodeB.filters[0]) > 0:

        newFilterShape = list(layerNodeB.filters[0].shape)
        if newFilterShape[0] > 0:
            newFilterShape[0] -= 1

        auxFilter = np.zeros((2, newFilterShape[0]), dtype=float)

        #auxFilter[0] = deleteLastFilter(layerNodeB.filters[0], newFilterShape)
        #auxFilter[1] = deleteLastFilter(layerNodeB.filters[1], newFilterShape)

        layerNodeB.filters = auxFilter

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