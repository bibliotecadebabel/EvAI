import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane
from DNA_creators import Creator as Creator_full

class DNA_Graph():

    def key2node(self,key):
        return self.graph.key2node[key]

    def node2key(self,node):
        q=node.objects[0]
        return q.shape

    def imprimir(self):
        for node in self.objects:
            print('The kid(s) of')
            print(self.node2key(node))
            print('are(is):')
            for nodek in node.kids:
                print(self.node2key(nodek))

    def length(self):
        return len(self.objects)

    def update(self):
        pass


    def __init__(self,center,size,dim,condition,
                typos=(0,(0,1,0,0)),type_add_layer=None,
                creator=Creator_full,
                num_morphisms=None,
                selector=None):
        self.typos = typos
        self.center=center
        self.x_dim=dim[0]
        self.y_dim=dim[1]
        self.size=size
        self.objects=None
        self.graph=None
        self.condition=condition
        self.creator=creator(self.typos,condition,type_add_layer)
        if num_morphisms:
            print('The selector is')
            print(selector)
            print('The creator is')
            print(creator)
            self.creator.num_morphisms=num_morphisms
            self.creator.Selector=selector
        g=self.creator.create(self.center,size)
        self.graph=g
        self.objects=list(g.key2node.values())
        self.version = type_add_layer
        self.node_max_particle=None









#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
