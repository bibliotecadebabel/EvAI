import numpy as np

class Node:
    def __init__(self, Value, Derivative, type):
        self.Value = Value
        self.Der = Derivative
        self.input_der=[]
        self.total_der=[]
        self.parents = []
        self.kids = []
        self.output=[]
        self.type = type

    def Fpro(self):
        pass
    def Bpro(self):
        self.Bpro.Bpro()

class Function_dot:
     def __init__(self,Node):
         self.Node=Node
     def Fpro(self):
        self.Node.parents=1
