class TestEntity():

    def __init__(self):
        self.id = 0
        self.name = None
    
    def load(self, data):
        self.id = data[0]
        self.name = data[1]