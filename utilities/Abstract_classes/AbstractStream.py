from abc import ABC, abstractmethod
import utilities.Node as nd
import utilities.P_trees as tr
from utilities.Safe import safe_append, safe_remove, safe_ope
import utilities.Graphs as gr

class Charge_log(ABC):
    def __init__(self):
        self.log = []
        self.signal=False

    def pop(self):
        if self.log:
            del self.log[0]

    def Currentvalue(self):
        if self.log:
            self.signal=True
            return self.log[0]
        else:
            return  None

    def charge(self,charge):
        self.log.extend(charge)

class Stream(ABC):

    def __init__(self, log_creator: Charge_log):
        self.Graph = gr.Graph()
        self.log_creator=log_creator

    @abstractmethod
    def charge_node(self,key):
        pass

    @abstractmethod
    def sync(self):
        pass

    def charge_nodes(self):
        keys = self.Graph.key2node.keys()
        for key in keys:
            self.charge_node(key)
        self.sync()

    def key2node(self,key):
        return self.Graph.key2node.get(key)

    def node2key(self,key):
        return self.Graph.node2key.get(key)

    def key2log(self,key):
        node = self.key2node(key)
        if node:
            return node.get_object()
        else:
            return node

    def add_node(self, key):
        node=nd.Node()
        Graph=self.Graph
        Graph.add_node(key,node)
        node.attach(self.log_creator())

    def remove_node(self, key):
        self.Graph.remove_node(key)

    def findCurrentvalue(self, key):
        node=self.key2node(key)
        if not(node):
            return node
        else:
            log=node.get_object()
            return log.Currentvalue()

    def pop_node(self,node):
        log=node.get_object()
        if log.log:
            log.pop()
        else:
            key=self.node2key(node)
            self.remove_node(key)

    def pop(self):
        for node in list(self.Graph.key2node.values()):
            self.pop_node(node)
