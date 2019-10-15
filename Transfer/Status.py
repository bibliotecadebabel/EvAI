class Status():
    def __init__(self):
        self.dt = 0.01
        self.tau=0.005
        self.n = 10
        self.r=3
        self.dx = 2
        self.L = 1
        self.beta = 2
        self.alpha = 1
        self.active = False
        self.particles_transfer = []
    
    def jsonToObject(self, body):

        self.dt = body['dt']
        self.tau = body['tau']
        self.n = body['n']
        self.r = body['r']
        self.dx = body['dx']
        self.L = body['L']
        self.beta = body['beta']
        self.alpha = body['alpha']
        self.active = body['active']
        self.particles_transfer = body['particles_transfer']