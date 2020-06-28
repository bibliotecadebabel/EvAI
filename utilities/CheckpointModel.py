class CheckPointModel():

    def __init__(self, alaiTime, dna):

        self.alaiTime = alaiTime
        self.dna = dna
    
    def formatDNA(self):
        
        if isinstance(self.dna, list):
            try:
                for i in range(len(self.dna)):
                    self.dna[i] = tuple(self.dna[i])
                
                self.dna = tuple(self.dna)
            except:
                print("INVALID DNA FORMAT: ", self.dna)
                raise