import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane


class Creator_from_selection_duplicate():

    def __init__(self,typos,condition,type_add_layer=None):
        if type_add_layer==None:
            from DNA_directions import directions
        elif type_add_layer=='inclusion':
            from DNA_directions_i import directions,directions_labels
            self.directions_labels=directions_labels
        else:
            from DNA_directions_duplicate import directions,directions_labels
            self.directions_labels=directions_labels
        self.typos=typos
        self.condition=condition
        self.directions=directions
        self.type_add_layer=type_add_layer


    def add_node(self,g,DNA):
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)

    def node2plane(self,node):
        q=node.objects[0]
        p=q.objects[0]
        return p

    def create(self,center,size,g=None):
        condition=self.condition
        if size>0:
            if isinstance(center,tuple):
                g=gr.Graph()
                self.add_node(g,center)
                center=g.key2node.get(center)
                self.create(center,size,g)
            else:
                q=center.objects[0]
                DNA_o=q.shape
                node_o=center
                for typo in self.typos:
                    if DNA_o:
                        for k in range (len(DNA_o)-2):
                            if k == typo[0]:
        #                       print('The value of k is')
        #                       print(k)
        #                       print('The value of DNA_o is:')
        #                       print(DNA_o)
                                direction=self.directions.get(typo[1])
                                if direction:
                                    DNA_f=condition(direction(k,DNA_o))
                                    if DNA_f:
                                        self.add_node(g,DNA_f)
                                        node_f=g.key2node.get(DNA_f)
                                        g.add_edges(DNA_o,[DNA_f])
                                        if self.type_add_layer:
                                            label=typo
                                            node=g.key2node.get(DNA_f)
                                            p=self.node2plane(node)
                                            if not (p.direction):
                                                p.direction=label
                if center.kids:
                    for kid in center.kids:
                        self.create(kid,size-1,g)
        return g
