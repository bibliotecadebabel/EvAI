from utilities.Abstract_classes.AbstractStream import Stream

class TorchStream(Stream):
    def __init__(self,nets,iterations,dt):
        super().__init__()
    Instruct=TorchInstruct(iterations,dt)
    self.Selector=TorchSelector(Instruct,net)
    def charge_node(self,key,charger=None):
        net=charger[0]
        Instruct=charger[1]
        node=self.Graph.key2node[key]
        log=Instruct.net2energy(net)
        node.objects.append(log)

    class TorchSelector():
        def __init__(self,Instruct,nets):
            self.Instruct=Instrut
            self.parents=nets
        def key2charger(self,key):
            u=[]
            u.append(self.nets[key])
            u.append(Instruct)
            return u

    class TorchInstruct():
        def __init__(self,iterations,dt):
            self.iterations=iterations
            self.dt=dt
        def net2energy(self,key):
            u=None
            return u
