from utilities.Abstract_classes.AbstractStream import Stream, Charge_log

class TorchStream(Stream):
    class Torch_log_creator(Charge_log):
        def __init__(self):
            super().__init__()
            self.old_net=None
            self.new_net=None
            self.plane=None
            self.log_size=None
    def __init__(self,log_size):
        super().__init__(self.Torch_log_creator)
        self.log_size=log_size

    def charge_node(self,key):
        log=self.key2log(key)
        log.charge([0]*self.log_size)

    def link_node(self,key,plane,net=None):
        log=self.key2log(key)
        log.plane=plane
        log.log_size=self.log_size
        if net is not None:
            log.old_net=net
            log.new_net=net
        return

    def sync(self):
        pass



    """def charge_node(self,key,charger=None):
        node=self.Graph.key2node[key]
        node.objects.append(0)"""
