import json
from DNA_Graph import DNA_Graph

class Controller:

    def __init__(self, dna_graph):

        self.dna_graph = dna_graph

    def __readFullFile(self, fileName):

        data = None
        try:
            file = open(fileName+".txt", "r")
            data = file.read()
            file.close()
        
        except Exception:
            pass
            
        return data
    
    def __writeFile(self, data, fileName):

        file = open(fileName+".txt",'w')
        file.write(str(json.dumps(data.__dict__)))
        file.close()

    def writeTangent(self, idNumber, fileName):

        data = None
        dataFile = DataFile()
        graphDTO = DNAGraphTangentDTO(self.dna_graph)

        data = self.__readFullFile(fileName)

        if data is not None and len(data) > 1:
            data = json.loads(data)
            dataFile.jsonToObject(data)
        
        value = str(json.dumps(graphDTO.__dict__))
        dataFile.dictionary[str(idNumber)] = None
        dataFile.dictionary[str(idNumber)] = value

        self.__writeFile(dataFile, fileName)


    def writeGraphShape(self, idNumber, fileName):
        data = None
        dataFile = DataFile()
        graphDTO = DNAGraphShapeDTO(self.dna_graph)

        data = self.__readFullFile(fileName)

        if data is not None and len(data) > 1:
            data = json.loads(data)
            dataFile.jsonToObject(data)
        
        value = str(json.dumps(graphDTO.__dict__))
        dataFile.dictionary[str(idNumber)] = None
        dataFile.dictionary[str(idNumber)] = value

        self.__writeFile(dataFile, fileName)

    def loadGraphShape(self, idNumber, fileName):

        dataFile = DataFile()
        data = self.__readFullFile(fileName)

        value = None

        if data is not None and len(data) > 1:
            data = json.loads(data)
            dataFile.jsonToObject(data)

            dto = DNAGraphShapeDTO(None)

            graph = dataFile.dictionary[str(idNumber)]

            graph = json.loads(graph)
            dto.jsonToObject(graph)

            value = dto.dictionary
        
        else:
            print("FILE NOUT FOUND (",fileName,")")

        return value

        

    
    '''
    def cargar(self, idNumber, fileName):

        data = None
        graph = None
        dataFile = DataFile()
        value = None

        try:
            file = open(fileName+".txt", "r")
            data = file.read()
            file.close()
        
        except Exception:
            print("archivo no existe= ",fileName+".txt")

        if data is not None and len(data) > 1:
            data = json.loads(data)
            dataFile.jsonToObject(data)
            value = dataFile.dictionary[str(idNumber)]

        if value is not None:
            
            graph = GraphDTO.GraphDTO()
            graph.jsonToObject(json.loads(value))
        
        return graph
    '''
class DNAGraphTangentDTO():
    
    def __init__(self, dna_graph):

        self.dictionary = {}
        
        if dna_graph is not None:

            for node in dna_graph.objects:
                key = str(node.objects[0].shape) # quadrant
                tangentPlane = node.objects[0].objects[0] # tangentPlane
                value = TangentPlaneDTO(tangentPlane)
                self.dictionary[key] = None
                self.dictionary[key] = str(json.dumps(value.__dict__))

class DNAGraphShapeDTO():
    
    def __init__(self, dna_graph):

        self.dictionary = {}

        if dna_graph is not None:

            for node in dna_graph.objects:
                key = str(node.objects[0].shape) # quadrant
                value = []

                for nodeKid in node.kids:
                    kidShape = str(nodeKid.objects[0].shape) # quadrant kid
                    value.append(kidShape)
                
                self.dictionary[key] = None
                self.dictionary[key] = value

    def jsonToObject(self, data):   
        
        self.dictionary = data["dictionary"]



class TangentPlaneDTO():

    def __init__(self, tangentPlane):

        self.divergence=tangentPlane.divergence
        self.metric=tangentPlane.metric
        self.density=tangentPlane.density
        self.num_particles=tangentPlane.num_particles
        self.gradient=tangentPlane.gradient
        self.reg_density=tangentPlane.reg_density
        self.interaction_field=tangentPlane.interaction_field
        self.difussion_field=tangentPlane.difussion_field
        self.external_field=tangentPlane.external_field
        self.force_field=tangentPlane.force_field
        self.energy=tangentPlane.energy
        self.interaction_potential=tangentPlane.interaction_potential
        self.velocity_potential=tangentPlane.velocity_potential
        self.direction=tangentPlane.direction

class DataFile():

    def __init__(self):
        self.dictionary = {}
    
    def jsonToObject(self, data):
        self.dictionary = data["dictionary"]
            