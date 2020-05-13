import json
from Entities.TangentPlaneEntity import TangentPlaneEntity

class TestModelEntity():

    def __init__(self):
        self.id = 0
        self.idTest = 0
        self.iteration = 0
        self.dna = None
        self.model_name = None
        self.model_weight = None
    
    def load(self, data):

        self.id = data[0]
        self.idTest = data[1]
        self.iteration = data[2]
        self.dna = data[3]
        self.model_name = data[4]
        self.model_weight = data[5]

        self.__tupleDna()
    
    def __tupleDna(self):
        
        for i in range(len(self.dna)):
            self.dna[i] = tuple(self.dna[i])
        
        self.dna = tuple(self.dna)

