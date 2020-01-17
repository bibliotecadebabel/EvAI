import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane

class DNA_Graph():
    def key2node(self,key):
        return self.graph.key2node[key]

    def __init__(self,center,size):
        self.type = (0,(1,0,0,0))
        self.center=center
        self.x_dim=self.center[0][3]
        self.y_dim=self.center[0][4]
        self.size=size
        self.objects=None
        self.graph=None


        if self.type==(0,(1,0,0,0)):
            def create_DNA(filters):
                k_0=self.center[0][2]
                ks=[filters+k_0]
                x = self.x_dim
                y = self.y_dim
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
            while k<self.size:
                add_node(g,k+1)
                g.add_edges(create_DNA(k),[create_DNA(k+1)])
                k=k+1
            self.objects=list(g.key2node.values())
            self.graph=g

            print('hello')
            print(create_DNA(0))







#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
