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
        stream=self.stream
        stream.add_net(key)
        pass

    def node2V(self,node):
        stream=self.stream
        k_o=self.node2key(node)
        V_o=stream.findCurrentvalue(k_o)
        return V_o

    def node2net(self,node):
        stream=self.stream
        return stream.get_net(self.node2key(node))

    def mutate(self,node_o,node_f):
        k_f=self.node2key(node_f)
        stream=self.stream
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
        stream=self.stream
        nodes=self.objects
        for node in self.support:
            V_o=self.node2V(node)
            if V_o:
                p=self.node2plane(node)
                external_field=[]
                for kid in node.kids:
                    V_f=self.node2V(kid)
                    if not(V_f):
                        self.mutate(node,kid)
                        V_f=self.node2V(kid)
                    dV=V_f-V_o
                    key_f=self.node2key(kid)
                    stream.key2signal_on(key_f)
                    external_field.append(dV)
                key_o=key_f=self.node2key(node)
                stream.key2signal_on(key_o)
                p.external_field=external_field

    def update_interation_field(self):
        pass

    def update(self):
        #stream=self.Stream
        #stream.charge_nodes()
        self.DNA_graph.update()
        self.update_density()
        self.update_diffussion_field()
        self.update_external_field()
        self.update_interation_field()
        self.stream.pop()




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
        self.stream=TorchStream(dataGen,10)






#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
