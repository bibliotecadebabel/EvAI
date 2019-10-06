import net2.Functions as Functions

# Node.objects[0] = Layer
class Layer():

    def __init__(self, node):
        self.filters = None
        self.node = node
        self.propagate = None
        self.value = None
        self.label = None
        self.bias = None
        self.backPropagate = None

        self.value_der = None
        self.filter_der = None
        self.bias_der = None

        self.value_der_total = None
        self.filter_der_total = None
        self.bias_der_total = None


def createLayerA(nodeA, objects):
    layer = Layer(nodeA)

    layer.propagate = Functions.Nothing
    layer.backPropagate= Functions.a_filter_der

    layer.filters = Functions.createTensorRandom((objects[2], objects[0], objects[1], 3))
    layer.bias = Functions.createTensorRandom((objects[0], objects[1], 3))
    layer.value = Functions.createTensorRandom((objects[0], objects[1], 3))

    layer.filter_der = Functions.createTensorRandom(layer.filters.shape)
    layer.filter_der_total = Functions.createTensorRandom(layer.filters.shape)

    layer.value_der = Functions.createTensorRandom(layer.value.shape)
    layer.value_der_total = Functions.createTensorRandom(layer.value.shape)

    layer.bias_der = Functions.createTensorRandom(layer.bias.shape)
    layer.bias_der_total = Functions.createTensorRandom(layer.bias.shape)

    return layer

def createLayerB(nodeB, objects):

    layer = Layer(nodeB)
    layerParent = nodeB.parents[0].objects[0]

    layer.propagate = Functions.ProductoPunto
    layer.backPropagate = Functions.b_filter_der

    layer.filters = Functions.createTensorRandom((2, objects[2]))
    layer.filter_der = Functions.createTensorRandom(layer.filters.shape)
    layer.filter_der_total = Functions.createTensorRandom(layer.filters.shape)


    layer.value = Functions.createTensorRandom(len(layerParent.filters))
    layer.value_der = Functions.createTensorRandom(layer.value.shape)
    layer.value_der_total = Functions.createTensorRandom(layer.value.shape)


    layer.bias = Functions.createTensorRandom(layer.value.shape)
    layer.bias_der = Functions.createTensorRandom(layer.value.shape)
    layer.bias_der_total = Functions.createTensorRandom(layer.value.shape)

    return layer

def createLayerC(nodeC):

    layer = Layer(nodeC)
    layerParent = nodeC.parents[0].objects[0]

    layer.propagate = Functions.ProductoPunto
    layer.backPropagate = Functions.c_filter_der

    layer.value = Functions.createTensorRandom((len(layerParent.filters)))
    layer.value_der = Functions.createTensorRandom(layer.value.shape)
    layer.value_der_total = Functions.createTensorRandom(layer.value.shape)


    layer.bias = Functions.createTensorRandom(layer.value.shape)
    layer.bias_der = Functions.createTensorRandom(layer.value.shape)
    layer.bias_der_total = Functions.createTensorRandom(layer.value.shape)

    layer.label = "c"
    
    return layer

def createLayerD(nodeD):

    layer = Layer(nodeD)

    layer.propagate = Functions.probability
    layer.backPropagate = Functions.probability_der

    layer.value = Functions.generateNumberRandom()
    layer.value_der = Functions.generateNumberRandom()
    layer.value_der_total = Functions.generateNumberRandom()

    layer.bias = Functions.generateNumberRandom()
    layer.bias_der = Functions.generateNumberRandom()
    layer.bias_der_total = Functions.generateNumberRandom()

    return layer

def createLayerE(nodeE):

    layer = Layer(nodeE)

    layer.propagate = Functions.logaritmo
    layer.backPropagate = Functions.Nothing
    
    layer.value = Functions.generateNumberRandom()
    layer.value_der = Functions.generateNumberRandom()
    layer.value_der_total = Functions.generateNumberRandom()

    layer.bias = Functions.generateNumberRandom()
    layer.bias_der = Functions.generateNumberRandom()
    layer.bias_der_total = Functions.generateNumberRandom()

    return layer

def addFilterNodeA(layerNodeA):

    layerNodeB = layerNodeA.node.kids[0].objects[0]

    newFilterShape = list(layerNodeA.filters.shape)
    newFilterShape[0] += 1

    layerNodeA.filters = Functions.createNewTensor(layerNodeA.filters, newFilterShape)

    layerNodeA.filter_der = Functions.createNewTensor(layerNodeA.filter_der, newFilterShape)

    layerNodeA.filter_der_total = Functions.createNewTensor(layerNodeA.filter_der_total, newFilterShape)

    newValueKidShape = [len(layerNodeA.filters)]
    
    layerNodeB.value = Functions.createNewTensor(layerNodeB.value, newValueKidShape)
    layerNodeB.value_der = Functions.createNewTensor(layerNodeB.value_der, newValueKidShape)
    layerNodeB.value_der_total = Functions.createNewTensor(layerNodeB.value_der_total, newValueKidShape)

    layerNodeB.bias = Functions.createNewTensor(layerNodeB.bias, newValueKidShape)
    layerNodeB.bias_der = Functions.createNewTensor(layerNodeB.bias_der, newValueKidShape)
    layerNodeB.bias_der_total = Functions.createNewTensor(layerNodeB.bias_der_total, newValueKidShape)


def deleteFilterNodeA(layerNodeA):
    
    layerNodeB = layerNodeA.node.kids[0].objects[0]
    
    newShape = list(layerNodeA.filters.shape)

    if newShape[0] > 0:
        newShape[0] -= 1

    layerNodeA.filters = Functions.deleteLastTensorDimension(layerNodeA.filters, newShape)
    layerNodeA.filter_der = Functions.deleteLastTensorDimension(layerNodeA.filter_der, newShape)
    layerNodeA.filter_der_total = Functions.deleteLastTensorDimension(layerNodeA.filter_der_total, newShape)

    newKidValueShape = [len(layerNodeA.filters)]

    layerNodeB.value = Functions.deleteLastTensorDimension(layerNodeB.value, newKidValueShape)
    layerNodeB.value_der = Functions.deleteLastTensorDimension(layerNodeB.value_der, newKidValueShape)
    layerNodeB.value_der_total = Functions.deleteLastTensorDimension(layerNodeB.value_der_total, newKidValueShape)

    layerNodeB.bias = Functions.deleteLastTensorDimension(layerNodeB.bias, newKidValueShape)
    layerNodeB.bias_der = Functions.deleteLastTensorDimension(layerNodeB.bias_der, newKidValueShape)
    layerNodeB.bias_der_total = Functions.deleteLastTensorDimension(layerNodeB.bias_der_total, newKidValueShape)

def addFilterNodeB(layerNodeB):

    newFilterShape = list(layerNodeB.filters[0].shape)
    newFilterShape[0] += 1

    auxFilter = Functions.createTensorRandom((2, newFilterShape[0]))
    auxFilter_der = Functions.createTensorRandom((2, newFilterShape[0]))
    auxFilter_der_total = Functions.createTensorRandom((2, newFilterShape[0]))

    auxFilter[0] = Functions.createNewTensor(layerNodeB.filters[0], newFilterShape)
    auxFilter[1] = Functions.createNewTensor(layerNodeB.filters[1], newFilterShape)

    auxFilter_der[0] = Functions.createNewTensor(layerNodeB.filter_der[0], newFilterShape)
    auxFilter_der[1] = Functions.createNewTensor(layerNodeB.filter_der[1], newFilterShape)

    auxFilter_der_total[0] = Functions.createNewTensor(layerNodeB.filter_der_total[0], newFilterShape)
    auxFilter_der_total[1] = Functions.createNewTensor(layerNodeB.filter_der_total[1], newFilterShape)

    layerNodeB.filters = auxFilter
    layerNodeB.filter_der = auxFilter_der
    layerNodeB.filter_der_total = auxFilter_der_total

def deleteFilterNodeB(layerNodeB):

    newShape = list(layerNodeB.filters[0].shape)

    if newShape[0] > 0:
        newShape[0] -= 1
    
    auxFilter = Functions.createTensorRandom((2, newShape[0]))
    auxFilter_der = Functions.createTensorRandom((2, newShape[0]))
    auxFilter_der_total = Functions.createTensorRandom((2, newShape[0]))


    auxFilter[0] = Functions.deleteLastTensorDimension(layerNodeB.filters[0], newShape)
    auxFilter[1] = Functions.deleteLastTensorDimension(layerNodeB.filters[1], newShape)

    auxFilter_der[0] = Functions.deleteLastTensorDimension(layerNodeB.filter_der[0], newShape)
    auxFilter_der[1] = Functions.deleteLastTensorDimension(layerNodeB.filter_der[1], newShape)

    auxFilter_der_total[0] = Functions.deleteLastTensorDimension(layerNodeB.filter_der_total[0], newShape)
    auxFilter_der_total[1] = Functions.deleteLastTensorDimension(layerNodeB.filter_der_total[1], newShape)

    layerNodeB.filters = auxFilter
    layerNodeB.filter_der = auxFilter_der
    layerNodeB.filter_der_total = auxFilter_der_total