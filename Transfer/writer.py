import time
import random
import Status as St
import json


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


