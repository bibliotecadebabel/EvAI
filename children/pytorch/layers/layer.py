
class Layer():

    def __init__(self, dna):
        
        self.dna = dna
        self.node = None
        self.value = None
        self.connected_layers = []