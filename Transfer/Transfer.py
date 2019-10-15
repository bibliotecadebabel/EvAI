from Transfer.Product_transfer import Transfer

class TransferLocal(Transfer):
    def __init__(self,status,name_read,name_write):
        super().__init__(status,name_read,name_write)

    def un_load(self):
        status=self.status
        transfer=self.status_transfer
        transfer.dt=status.dt
        transfer.n=status.n
        transfer.dx=status.dx
        transfer.beta=status.beta
        transfer.active = status.active
    def load(self):
        i=0
        transfer=self.status_transfer
        while  i<len(self.status.objects):
            node=self.status.objects[i]
            q=node.objects[0]
            p=q.objects[0]
            p.num_particles=transfer.particles[i]
            i=i+1




class TransferRemote(Transfer):

    def __init__(self,status,name_read,name_write):
        super().__init__(status,name_read,name_write)

    def un_load(self):
        transfer=self.status_transfer
        i=0
        transfer.particles=[]
        while  i<len(self.status.objects):
            node=self.status.objects[i]
            q=node.objects[0]
            p=q.objects[0]
            transfer.particles.append(p.num_particles)
            i=i+1

    def load(self):
        status=self.status
        transfer=self.status_transfer
        status.dt=transfer.dt
        status.n=transfer.n
        status.dx=transfer.dx
        status.beta=transfer.beta
        status.active = transfer.active



"""tran=TransferLocal('status','read','write','special')
transremote= TransferRemote()

tran.update()
transremote.update()"""

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
