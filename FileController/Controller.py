import json

class Controller:

    def __init__(self, dna_graph):

        self.dna_graph = dna_graph

    def imprimir(self, idNumber, fileName):

        data = None
        dataFile = DataFile()
        graphDTO = DNAGraphDTO(self.dna_graph)

        try:
            file = open(fileName+".txt", "r")
            data = file.read()
            file.close()
        
        except Exception:
            pass

        if data is not None and len(data) > 1:
            data = json.loads(data)
            dataFile.jsonToObject(data)
        
        value = str(json.dumps(graphDTO.__dict__))
        dataFile.dictionary[str(idNumber)] = None
        dataFile.dictionary[str(idNumber)] = value

        file = open(fileName+".txt",'w')
        file.write(str(json.dumps(dataFile.__dict__)))
        file.close()
    
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
class DNAGraphDTO():
    
    def __init__(self, dna_graph):

        self.dictionary = {}

        for node in dna_graph.objects:
            key = str(node.objects[0].shape) # quadrant
            tangentPlane = node.objects[0].objects[0] # tangentPlane
            value = TangentPlaneDTO(tangentPlane)
            self.dictionary[key] = str(json.dumps(value.__dict__))
    
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

class DataFile():

    def __init__(self):
        self.dictionary = {}
    
    def jsonToObject(self, data):
        self.dictionary = data["dictionary"]
            