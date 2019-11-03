from utilities.Abstract_classes.AbstractStream import Stream

class TestStream(Stream):
    def __init__(self):
        super().__init__()

    def charge_node(self,key,charger=None):
        node=self.Graph.key2node[key]
        node.objects.append(0)
