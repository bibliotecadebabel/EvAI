from abc import abstractmethod

class Layer():

    def __init__(self, adn):
        
        self.adn = adn
        self.object = None
        self.node = None
        self.value = None
        self.connected_layers = []
    
    def propagate(self):
        pass

    def deleteParam(self):
        pass

    def viewGraph(self):

        print("graph of: ", self.adn)
        for kid in self.node.kids:
            print("kid: ", kid.objects[0].adn)
        
        for parent in self.node.parents:
            print("parent: ", parent.objects[0].adn)