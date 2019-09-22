import Functions

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

    layer.filter_der = Functions.createTensorZero(layer.filters.shape)
    layer.filter_der_total = Functions.createTensorZero(layer.filters.shape)

    layer.value_der = Functions.createTensorZero(layer.value.shape)
    layer.value_der_total = Functions.createTensorZero(layer.value.shape)

    layer.bias_der = Functions.createTensorZero(layer.bias.shape)
    layer.bias_der_total = Functions.createTensorZero(layer.bias.shape)

    return layer

def createLayerB(nodeB, objects):

    layer = Layer(nodeB)
    layerParent = nodeB.parents[0].objects[0]

    layer.propagate = Functions.ProductoPunto
    layer.backPropagate = Functions.b_filter_der

    layer.filters = Functions.createTensorRandom((2, objects[2]))
    layer.filter_der = Functions.createTensorZero(layer.filters.shape)
    layer.filter_der_total = Functions.createTensorZero(layer.filters.shape)


    layer.value = Functions.createTensorZero((len(layerParent.filters)))
    layer.value_der = Functions.createTensorZero(layer.value.shape)
    layer.value_der_total = Functions.createTensorZero(layer.value.shape)


    layer.bias = Functions.createTensorZero(layer.value.shape)
    layer.bias_der = Functions.createTensorZero(layer.value.shape)
    layer.bias_der_total = Functions.createTensorZero(layer.value.shape)

    return layer

def createLayerC(nodeC):

    layer = Layer(nodeC)
    layerParent = nodeC.parents[0].objects[0]

    layer.propagate = Functions.ProductoPunto
    layer.backPropagate = Functions.c_filter_der

    layer.value = Functions.createTensorZero((len(layerParent.filters)))
    layer.value_der = Functions.createTensorZero(layer.value.shape)
    layer.value_der_total = Functions.createTensorZero(layer.value.shape)


    layer.bias = Functions.createTensorZero(layer.value.shape)
    layer.bias_der = Functions.createTensorZero(layer.value.shape)
    layer.bias_der_total = Functions.createTensorZero(layer.value.shape)

    return layer

def createLayerD(nodeD):

    layer = Layer(nodeD)

    layer.propagate = Functions.probability
    layer.backPropagate = Functions.probability_der

    return layer

def createLayerE(nodeE):

    layer = Layer(nodeE)

    layer.propagate = Functions.logaritmo
    layer.backPropagate = Functions.Nothing

    return layer



