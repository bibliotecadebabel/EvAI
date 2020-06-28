import json
from Entities.TangentPlaneEntity import TangentPlaneEntity

class TestModelEntity():

#current_time, current_alai_time, reset_dt_count, type
    def __init__(self):
        self.id = 0
        self.idTest = 0
        self.iteration = 0
        self.dna = None
        self.model_name = None
        self.model_weight = None
        self.current_time = 0
        self.current_alai_time = 0
        self.reset_dt_count = 0
        self.type = None
    
    def load(self, data):

        self.id = data[0]
        self.idTest = data[1]
        self.dna = data[2]
        self.iteration = data[3]
        self.model_name = data[4]
        self.model_weight = data[5]
        self.current_time = float(data[6])
        self.current_alai_time = float(data[7])
        self.reset_dt_count = data[8]
        self.type = data[9]

        #self.__tupleDna()
    
    def __tupleDna(self):
        
        for i in range(len(self.dna)):
            self.dna[i] = tuple(self.dna[i])
        
        self.dna = tuple(self.dna)

