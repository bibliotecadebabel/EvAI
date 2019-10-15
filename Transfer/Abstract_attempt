from abc import ABC, abstractmethod


class Transfer(ABC):
    def __init__(self,status):
        super().__init__()
        self.status=status

    @abstractmethod
    def write(self):
        pass

    @abstractmethod
    def read(self):
        pass

print('hi')
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
