import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane
from DNA_conditions import Mutations

class creator(types,condition):
    self.types=types
    self.condition=condition
    self.directions=[]
    for type in types:
        direction=Mutations.get(type)
        if direction:
            self.directions.append(direction)
            type=tuple(i * (-1) for i in type)
            direction=Mutations.get(type)
            if direction:
                self.directions.append(direction)

    def add_node(g,DNA):
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)

    def create(self,center,size,g=None):
        if not(size<0):
            if type(center) == tuple:
                g=gr.Graph()
                add_node(g,center)
                center=g.key2node.get(center)
                create(self,center,size,g)
            else:
                q=center.objects[0]
                DNA_o=q.shape
                node_o=center
                for direction in self.directions:
                    DNA_f=condition(direction(DNA_o))
                    if DNA_f:
                        add_node(g,DNA_f)
                        node_f=g.key2node.get(DNA_f)
                        g.add_edge(DNA_o,DNA_f)
                for kid in center.kids:
                    create(self,kid,size-1,g)
        return g




class Direction():
    def __init__(self,type,creator):
        self.type=type
        self.creator=creator


Mutations=[]

#linear graph that changes value increases x,y dimension of kernel

type=(0,(0,1,0,0))

def increase_kernel_width(DNA_graph):
    def create_DNA(width):
        k_0=DNA_graph.center[0][3]+width
        filters=DNA_graph.center[0][2]
        x = DNA_graph.x_dim
        y = DNA_graph.y_dim
        networkADN = ((0, 3, filters, k_0, k_0),
            (0, 1, int(filters/2), x-k_0+1,y-k_0+1),
            (1, int(filters/2), 2), (2,))
        return networkADN
    def add_node(g,i):
        DNA=create_DNA(i)
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)
    g=gr.Graph()
    add_node(g,0)
    k=0
    while k<DNA_graph.size:
        add_node(g,k+1)
        g.add_edges(create_DNA(k),[create_DNA(k+1)])
        k=k+1
    return g

creator=increase_kernel_width
Mutations.append(Direction(type,creator))



#same as above but with only 3D convolutions


type=(0,(0,0,0,1,1))


def increase_kernel_width(DNA_graph):
    print('Accesssed createor')
    def create_DNA(width):
        k_0=DNA_graph.center[0][3]+width+1
        filters=DNA_graph.center[0][2]
        x = DNA_graph.x_dim
        y = DNA_graph.y_dim
        networkADN = ((0, 3, filters,1, k_0, k_0),
            (0, 1, int(filters/2),1,x-k_0+1,y-k_0+1),
            (1, int(filters/2), 2), (2,))
        return networkADN
    def add_node(g,i):
        DNA=create_DNA(i)
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)
    g=gr.Graph()
    add_node(g,0)
    k=0
    while k<DNA_graph.size:
        add_node(g,k+1)
        g.add_edges(create_DNA(k),[create_DNA(k+1)])
        k=k+1
    return g

creator=increase_kernel_width
Mutations.append(Direction(type,creator))


type=(0,(0,0,1,0,0))

def increase_kernel_depth(DNA_graph):
    def create_DNA(width):
        width_0=DNA_graph.center[1][3]
        width=width+width_0
        k_0=DNA_graph.center[0][4]
        out_filters=DNA_graph.center[1][2]
        outer_filters=DNA_graph.center[2][2]
        filters=DNA_graph.center[0][2]
        x = DNA_graph.x_dim
        y = DNA_graph.y_dim
        networkADN = ((0, 3, filters,1, k_0, k_0),
            (0,1,out_filters,width,3,3),
            (0,1,outer_filters,(filters-width+1)*out_filters,x-2*k_0+2,x-2*k_0+2),
            (1,outer_filters, 2), (2,))
        return networkADN
    def add_node(g,i):
        DNA=create_DNA(i)
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)
    g=gr.Graph()
    add_node(g,0)
    k=0
    while k<DNA_graph.size:
        add_node(g,k+1)
        g.add_edges(create_DNA(k),[create_DNA(k+1)])
        k=k+1
    return g

creator=increase_kernel_depth
Mutations.append(Direction(type,creator))


#linear graph that adds filters, same as below

type=(0,(0,1,0,0,0))
def add_filter_new(DNA_graph):
    def create_DNA(filters):
        k_0=DNA_graph.center[0][2]
        ks=[filters+k_0]
        x = DNA_graph.x_dim
        y = DNA_graph.y_dim
        networkADN = ((0, 3, ks[0],1, x, y), (1, ks[0], 2), (2,))
        return networkADN
    def add_node(g,i):
        DNA=create_DNA(i)
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)
    g=gr.Graph()
    add_node(g,0)
    k=0
    k=0
    while k<DNA_graph.size:
        add_node(g,k+1)
        g.add_edges(create_DNA(k),[create_DNA(k+1)])
        k=k+1
    return g

creator=add_filter_new

Mutations.append(Direction(type,creator))


#linear graph that adds filters

type=(0,(1,0,0,0))
def add_filter(DNA_graph):
    def create_DNA(filters):
        k_0=DNA_graph.center[0][2]
        ks=[filters+k_0]
        x = DNA_graph.x_dim
        y = DNA_graph.y_dim
        networkADN = ((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
        return networkADN
    def add_node(g,i):
        DNA=create_DNA(i)
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)
    g=gr.Graph()
    add_node(g,0)
    k=0
    k=0
    while k<DNA_graph.size:
        add_node(g,k+1)
        g.add_edges(create_DNA(k),[create_DNA(k+1)])
        k=k+1
    return g

creator=add_filter

Mutations.append(Direction(type,creator))

Directions={}

for mutation in Mutations:
    Directions.update({mutation.type : mutation.creator})
