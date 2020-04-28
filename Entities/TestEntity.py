class TestEntity():

    def __init__(self):
        self.id = 0
        self.name = None
        self.period = 0
        self.dt = 0
        self.total_iteration = 0
    
    def load(self, data):
        self.id = data[0]
        self.name = data[1]
        self.period = data[2]
        self.dt = data[3]
        self.total_iteration = data[4]