#import utilities.Quadrants as qu
#import utilities.Node as nd
#import utilities.Graphs as gr
#import TangentPlane as tplane
from Particle import particle as particle


class DNA_Phase_space():
    def key2node(self,key):
        return self.DNA_graph.key2node(key)

    def center(self):
        return self.DNA_graph.center

    def node2plane(self,node):
        q=node.objects[0]
        return q.objects[0]

    def create_particles(self,N,key=None):
        if key == None:
            key=self.center()
        k=0
        self.num_particles=N
        node=self.key2node(key)
        p=self.node2plane(node)
        while k<N:
            par=particle()
            par.position.append(node)
            par.velocity.append(node)
            #par.objects.append(log)
            p.particles.append(par)
            p.num_particles+=1
            k=k+1

    def update_density(self):
        nodes=self.objects
        N=self.num_particles
        for node in nodes:
            p=self.node2plane(node)
            p.density=p.num_particles/N

    def update_divergence(self):
        nodes=self.objects
        N=self.num_particles
        for node in nodes:
            p=self.node2plane(node)
            p.divergence=0
            for kid_node in node.kids:
                pf=self.node2plane(kid_node)
                p.divergence=p.divergence+(
                    pf.num_particles-p.num_particles)/N

    #It seems the current version cannot handle regularization and negarive
    #Pourus medium exponent

    def __init__(self,DNA_graph,
            Potential=None,Interaction=None,External=None):
        self.DNA_graph =DNA_graph
        self.objects=DNA_graph.objects
        self.num_particles=None
        print('Hi')






#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
