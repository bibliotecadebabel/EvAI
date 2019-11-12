from abc import ABC, abstractmethod
import utilities.Node as nd
import utilities.P_trees as tr
from utilities.Safe import safe_append, safe_remove, safe_ope
import utilities.Graphs as gr

class Stream(ABC):

    def __init__(self):
        self.Graph = gr.Graph()


    @abstractmethod
    def charge_node(self,key,charger=None):
        pass

    def charge_nodes(self,selector=None):
        keys = self.Graph.key2node.keys()
        if selector is None:
            for key in keys:
                self.charge_node(key)
        if not(selector==None):
            for key in keys:
                self.charge_node(key,selector(key))

    def pop_node(self,node):
        if not node.objects==[]:
            del node.objects[0]
        pass

    def pop(self):
        for node in list(self.Graph.key2node.values()):
            self.pop_node(node)
        pass

    def findCurrentvalue(self, key):
        if not((self.Graph.key2node[key]).objects==[]):
            return (self.Graph.key2node[key]).objects[0]

    def add_node(self, key):
        self.Graph.add_node(key,nd.Node())

    def remove_node(self, key):
        self.Graph.remove_node(key)
