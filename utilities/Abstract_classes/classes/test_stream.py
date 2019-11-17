from utilities.Abstract_classes.AbstractStream import Stream, Charge_log

class TestStream(Stream):
    class Test_log_creator(Charge_log):
        def __init__(self):
            super().__init__()
    def __init__(self):
        super().__init__(self.Test_log_creator)
    def charge_node(self,key):
        log=self.key2log(key)
        log.charge([0,2])
    def sync(self):
        pass



    """def charge_node(self,key,charger=None):
        node=self.Graph.key2node[key]
        node.objects.append(0)"""
