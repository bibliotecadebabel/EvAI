import json
from Entities.tangent_plane_entity import TangentPlaneEntity

class TestResultEntity():

    def __init__(self):
        self.id = 0
        self.idTest = 0
        self.iteration = 0
        self.dna = None
        self.tangentPlane = None
        self.isCenter = 0
        self.current_alai_time = 0
    
    def load(self, data):

        tangentPlane = TangentPlaneEntity()
        tangentPlane.load(json.loads(data[4]))

        self.id = data[0]
        self.idTest = data[1]
        self.iteration = data[2]
        self.dna = json.loads(data[3])['dna']
        self.tangentPlane = tangentPlane
        self.isCenter = data[5]
        self.current_alai_time = float(data[7])

        self.__tupleDna()
    
    def __tupleDna(self):
        
        for i in range(len(self.dna)):
            self.dna[i] = tuple(self.dna[i])
        
        self.dna = tuple(self.dna)

