
# Node.objects[0] = Layer
class Layer():

    def __init__(self, propagate, node, filters=None, value=None, bias=None, backPropagate=None):
        self.filters = filters
        self.node = node
        self.propagate = propagate
        self.value = value
        self.label = None
        self.bias = bias ## bias = filters
        self.backPropagate = backPropagate

        self.value_der = None
        self.filter_der = None
        self.bias_der = None

        self.value_der_total = None
        self.filter_der_total = None
        self.bias_der_total = None



                
        





