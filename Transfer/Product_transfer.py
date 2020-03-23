from abc import ABC, abstractmethod
import json

class Transfer(ABC):
    def __init__(self,status,name_write,name_read):
        self.status=status
        self.status_transfer=status_transfer()
        self.name_read=name_read
        self.name_write=name_write

    @abstractmethod
    def un_load(self):
        pass

    @abstractmethod
    def load(self):
        pass

    def readLoad(self):
        self.read()
        if (self.status_transfer is not None
            and self.status_transfer.particles is not None):
            self.load()
        else:
            print("NONE")

    def write(self):
        status=self.status_transfer
        value = json.dumps(status.__dict__)
        file = open(self.name_write,'w')
        file.write(str(value))
        file.close()

    def read(self):

        try:
            status = status_transfer()
            file = open(self.name_read,'r')
            value = file.read()
            status.jsonToObject(json.loads(value))
            file.close()
        except Exception:
            status = None

        if status is not None:
            self.status_transfer = status

    def update(self):
        self.un_load()
        self.write()
        self.readLoad()
        pass

class status_transfer:
    def __init__(self):
        self.dt = None
        self.tau = None
        self.n = None
        self.r = None
        self.dx = None
        self.L = None
        self.beta = None
        self.alpha = None
        self.active = None
        self.particles= None
        self.typos = None

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
        self.particles = body['particles']
        self.typos = body['typos']



"""
def createParticles(status):

    amount = random.randint(20, 100)
    for i in range(0, amount):
        status.particles_transfer.append(random.randint(20, amount))

while True:

    status = St.Status()
    createParticles(status)

    value = json.dumps(status.__dict__)
    file = open("body.txt",'w')
    file.write(str(value))
    file.close()
    time.sleep(10)
"""
