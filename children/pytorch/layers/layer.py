from abc import abstractmethod

class Layer():

    def __init__(self, dna):
        
        self.dna = dna
        self.object = None
        self.node = None
        self.value = None
        self.connected_layers = []
    
    def propagate(self):
        pass

    def deleteParam(self):
        pass

    def viewGraph(self):

        print("graph of: ", self.dna)
        for kid in self.node.kids:
            print("kid: ", kid.objects[0].dna)
        
        for parent in self.node.parents:
            print("parent: ", parent.objects[0].dna)