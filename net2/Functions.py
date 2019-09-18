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
    
    value = np.zeros(len(parent.objects[0].filters))
    bias = np.zeros(len(parent.objects[0].filters))

    for i in range(len(parent.objects[0].filters)):
        value[i] = Dot(parent.objects[0].filters[i], (parent.objects[0].value + parent.objects[0].bias))

    for i in range(len(parent.objects[0].filters)):
        bias[i] = Dot(parent.objects[0].filters[i], (parent.objects[0].value + parent.objects[0].bias))

    layer.value = value
    layer.bias = bias

def probability(layer):
    parent = layer.node.parents[0]
    sc =  parent.objects[0].value[0] 
    sn = parent.objects[0].value[1]

    if parent.objects[0].label is None or parent.objects[0].label == "c":
        layer.value = np.exp(sc)/(np.exp(sc) + np.exp(sn))
    elif parent.objects[0].label == "n":
        layer.value = (np.exp(sn)/(np.exp(sc) + np.exp(sn)))

def logaritmo(layer):
    parent = layer.node.parents[0]
    layer.value = np.log(parent.objects[0].value)*-1

## f = filter
## v = value
def Dot(f, v):

    y = f * v
    y = y.sum()
    y = y / len(f)

    return y 

############### FUNCIONES BACKPROPAGATE ###############

def probability_der(layer):
    layer.value_der = (1/layer.value)*-1

def c_filter_der(layer):
    kid = layer.node.kids[0]
    p = kid.objects[0].value
    sc = layer.value[0] 
    sn = layer.value[1]

    filter_der = np.zeros(2, dtype=float)

    if kid.objects[0].label == "c":
        filter_der[0] = p - (p*p)
        filter_der[1] = (-(p*p)*np.exp(sn))/np.exp(sc)
    else:
        filter_der[0] = (-(p*p)*np.exp(sc))/np.exp(sn)
        filter_der[1] = p - (p*p)
    
    layer.value_der = filter_der*kid.objects[0].value_der

    layer.value_der_total = np.zeros((layer.value_der.shape), dtype=float)

def b_filter_der(layer):
    kid = layer.node.kids[0]
    filter_der = np.zeros((layer.filters.shape), dtype=float)

    filter_der[0] = layer.value*kid.objects[0].value_der[0]
    filter_der[1] = layer.value*kid.objects[0].value_der[1]

    value_der = np.zeros((layer.value.shape), dtype=float)
    bias_der = np.zeros((layer.value.shape), dtype=float)

    value_der = (layer.filters[0] * kid.objects[0].value_der[0]) + (layer.filters[1] * kid.objects[0].value_der[1])
    bias_der = (layer.filters[0] * kid.objects[0].value_der[0]) + (layer.filters[1] * kid.objects[0].value_der[1])

    layer.filter_der = filter_der
    layer.filter_der_total = np.zeros((layer.filter_der.shape), dtype=float)
    
    layer.value_der = value_der
    layer.value_der_total = np.zeros((layer.value_der.shape), dtype=float)

    layer.bias_der = bias_der
    layer.bias_der_total = np.zeros((layer.bias_der.shape), dtype=float)

def a_filter_der(layer):

    filter_der = np.zeros((layer.filters.shape), dtype=float)
    bias_der = np.zeros((layer.value.shape), dtype=float)
    kid = layer.node.kids[0]

    for i in range(layer.filters.shape[0]):
        filter_der[i] = (layer.value * kid.objects[0].value_der[i])/len(layer.value)

    for i in range(layer.filters.shape[0]):
        bias_der += (layer.filters[i] * kid.objects[0].value_der[i])/len(layer.filters[i])
    
    layer.filter_der = filter_der
    layer.bias_der = bias_der

    layer.bias_der_total = np.zeros((layer.bias_der.shape), dtype=float)
    layer.filter_der_total = np.zeros((layer.filter_der.shape), dtype=float)

############### ELIMINAR FILTROS ###############

def removeFilters(layer):
    removeFilterNodeA(layer)
    removeFilterNodeB(layer.node.kids[0].objects[0])

def removeFilterNodeA(layerNodeA):
    if layerNodeA.filters is not None and len(layerNodeA.filters) > 0:
        
        newFilterShape = list(layerNodeA.filters.shape)
        
        if newFilterShape[0] > 0:
            newFilterShape[0] -= 1
        
        layerNodeA.filters = deleteLastFilter(layerNodeA.filters, newFilterShape)

def removeFilterNodeB(layerNodeB):
    if layerNodeB.filters[0] is not None and len(layerNodeB.filters[0]) > 0:

        newFilterShape = list(layerNodeB.filters[0].shape)
        if newFilterShape[0] > 0:
            newFilterShape[0] -= 1

        auxFilter = np.zeros((2, newFilterShape[0]), dtype=float)

        auxFilter[0] = deleteLastFilter(layerNodeB.filters[0], newFilterShape)
        auxFilter[1] = deleteLastFilter(layerNodeB.filters[1], newFilterShape)

        layerNodeB.filters = auxFilter

def deleteLastFilter(oldFilter, newShape):
    
    newFilter = np.zeros(tuple(newShape), dtype=float)

    for i in range(newShape[0]):
        newFilter[i] = oldFilter[i]

    return newFilter

############### AGREGAR FILTROS ###############

def addFilters(layer):
        
    addFilterNodeA(layer)
    addFilterNodeB(layer.node.kids[0].objects[0])

def addFilterNodeA(layerNodeA):

    if layerNodeA.filters is not None and len(layerNodeA.filters) > 0:

            # Obtengo la estructura del tensor del filtro original y lo convierto a lista mutable
            newFilterShape = list(layerNodeA.filters.shape)
            
            # Aumento la cantidad de filtros (Valor de K)
            newFilterShape[0] += 1

            layerNodeA.filters = createNewFilterNodeA(layerNodeA.filters, newFilterShape)

# Crear nuevos filtros para el nodo A
def createNewFilterNodeA(oldFilter, newShape):
    
    # Creo la nueva estructura del filtro
    newFilter = np.zeros(tuple(newShape), dtype=float)


    # Conservo los valores del filtro viejo
    for i in range(len(oldFilter)):
        newFilter[i] = oldFilter[i]

    # Creo los nuevos valores random para el nuevo filtro K + 1.
    newFilter[len(newFilter) - 1] = np.random.rand(*oldFilter.shape[1:])

    return newFilter

def addFilterNodeB(layerNodeB):

    if layerNodeB.filters[0] is not None and len(layerNodeB.filters[0]) > 0:

        newFilterShape = list(layerNodeB.filters[0].shape)
        newFilterShape[0] += 1

        auxFilter = np.zeros((2, newFilterShape[0]), dtype=float)

        auxFilter[0] = createNewFilterNodeB(layerNodeB.filters[0], newFilterShape)
        auxFilter[1] = createNewFilterNodeB(layerNodeB.filters[1], newFilterShape)

        layerNodeB.filters = auxFilter


# Crear nuevos filtros / bias para el nodo B
def createNewFilterNodeB(oldFilter, newShape):
    newFilter = np.zeros(tuple(newShape), dtype=float)

    for i in range(len(oldFilter)):
        newFilter[i] = oldFilter[i]
        
    newFilter[len(newFilter) - 1] = random.uniform(0, 1)

    return newFilter



############### ESTRUCTURA INICIAL NODOS A & B ###############

def createFilterA(networkObjects):

    filters = np.random.rand(networkObjects[2],networkObjects[0], networkObjects[1], 3)
    return filters

def createValueA(networkObjects):

    return np.random.rand(networkObjects[0], networkObjects[1], 3)

def createFilterB(networkObjects):

    filters = np.random.rand(2, networkObjects[2])

    return filters
