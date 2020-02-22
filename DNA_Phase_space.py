#import utilities.Quadrants as qu
#import utilities.Node as nd
#import utilities.Graphs as gr
#import TangentPlane as tplane
from Particle import particle as particle
from utilities.Abstract_classes.classes.torch_stream import TorchStream
from DAO import GeneratorFromImage
import children.pytorch.MutateNetwork as Mutate


class DNA_Phase_space():
    def key2node(self,key):
        return self.DNA_graph.key2node(key)

    def node2key(self,node):
        q=node.objects[0]
        return q.shape

    def center(self):
        return self.DNA_graph.center

    def node2plane(self,node):
        q=node.objects[0]
        return q.objects[0]

    def add_net(self,key):
        stream=self.Stream
        stream.add_net(key)
        pass

    def node2V(self,node):
        stream=self.Stream
        k_o=self.node2key(node)
        V_o=stream.findCurrentvalue(k_o)
        return V_o

    def node2net(self,node):
        stream=self.Stream
        return stream.get_net(self.node2key(node))

    def mutate(self,node_o,node_f):
        k_f=self.node2key(node_f)
        stream=self.Stream
        if not(stream.get_net(k_f)):
            net=self.node2net(node_o)
            net_f=Mutate.executeMutation(net,k_f)
            stream.add_node(k_f)
            stream.link_node(k_f,net_f)
            stream.charge_node(k_f)

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
        self.add_net(key)
        self.support.append(node)

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

    def update_diffussion_field(self):
        nodes=self.objects
        beta=self.beta
        for node in nodes:
            p=self.node2plane(node)
            difussion_field=[]
            for kid in node.kids:
                pf=self.node2plane(kid)
                dU=beta*(pf.density**(beta-1)
                    -p.density**(beta-1))
                difussion_field.append(dU)
            p.diffussion_field=difussion_field

    def update_external_field(self):
        stream=self.Stream
        nodes=self.objects
        for node in self.support:
            print('The size of the support is')
            print(len(self.support))
            print('scaning node')
            V_o=self.node2V(node)
            if V_o:
                print('node was not empty')
                p=self.node2plane(node)
                external_field=[]
                for kid in node.kids:
                    print('scaning kid')
                    print('The value of Vf is')
                    V_f=self.node2V(kid)
                    if not(V_f):
                        print('kid was empty')
                        self.mutate(node,kid)
                        V_f=self.node2V(kid)
                        print('The value of Vf is')
                        print(V_f)
                    dV=V_f-V_o
                    external_field.append(dV)
                p.external_field=external_field


    #It seems the current version cannot handle regularization and negarive
    #Pourus medium exponent

    def __init__(self,DNA_graph,
            Potential=None,Interaction=None,External=None):
        self.DNA_graph =DNA_graph
        self.objects=DNA_graph.objects
        self.num_particles=None
        self.beta=2
        self.support=[]
        dataGen = GeneratorFromImage.GeneratorFromImage(2, 100, cuda=False)
        dataGen.dataConv2d()
        self.Stream=TorchStream(dataGen,10)






#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
