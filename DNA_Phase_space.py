from Particle import particle as particle
from utilities.Abstract_classes.classes.torch_stream import TorchStream
from DAO import GeneratorFromImage
import children.pytorch.MutateNetwork as Mutate
import utilities.Graphs as gr
import utilities.P_trees as tr


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

    def interction_kernel(self,r):
        alpha=self.alpha
        return  -1*(r/1.5)**(alpha)

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

    def update_interaction_potential(self):
        for node in self.objects:
            p=self.node2plane(node)
            distance=p.distance
            p.interaction_potential=0
            for key in distance.keys():
                node_k=key.objects[0]
                p_k=self.node2plane(node_k)
                p.interaction_potential=(p.interaction_potential
                    +p_k.density
                        *self.interction_kernel(distance[key]))

    def update_interaction_field(self):
        self.update_interaction_potential()
        for node in self.objects:
            p_o=self.node2plane(node)
            W_o=p_o.interaction_potential
            interaction_field=[]
            k=0
            for kid in node.kids:
                p_f=self.node2plane(kid)
                W_f=p_f.interaction_potential
                interaction_field.append(W_f-W_o)
                k=k+1
            p_o.interaction_field=interaction_field


    def print_diffussion_field(self):
        for node in self.objects:
            print('The diffusion field between')
            print(self.node2key(node))
            print('and')
            k=0
            for nodek in node.kids:
                print(self.node2key(nodek))
                print('is')
                p=self.node2plane(node)
                print(p.diffussion_field[k])
                k=k+1

    def print_external_field(self):
        for node in self.objects:
            print('The diffusion field between')
            print(self.node2key(node))
            print('and')
            k=0
            for nodek in node.kids:
                print(self.node2key(nodek))
                print('is')
                p=self.node2plane(node)
                if p.external_field:
                    print(p.external_field[k])
                else:
                    print(None)
                k=k+1

    def print_interaction_field(self):
        for node in self.objects:
            print('The interaction field between')
            print(self.node2key(node))
            print('and')
            k=0
            for nodek in node.kids:
                print(self.node2key(nodek))
                print('is')
                p=self.node2plane(node)
                if p.interaction_field:
                    print(p.interaction_field[k])
                else:
                    print(None)
                k=k+1

    def print_force_field(self):
        for node in self.objects:
            print('The force between')
            print(self.node2key(node))
            print('and')
            k=0
            for nodek in node.kids:
                print(self.node2key(nodek))
                print('is')
                p=self.node2plane(node)
                if p.force_field:
                    print(p.force_field[k])
                else:
                    print(None)
                k=k+1

    def attach_balls(self):
        for node in self.objects:
            p=self.node2plane(node)
            p.ball=gr.spanning_tree(node,n=self.radius)
            p.distance=tr.tree_distances(p.ball)

    def update(self):
        stream=self.stream
        stream.charge_nodes()
        self.DNA_graph.update()
        self.update_density()
        self.update_diffussion_field()
        self.update_external_field()
        self.update_interaction_field()
        self.stream.pop()




    #It seems the current version cannot handle regularization and negarive
    #Pourus medium exponent

    def __init__(self,DNA_graph,
            Potential=None,Interaction=None,External=None):
        self.DNA_graph = DNA_graph
        self.objects = DNA_graph.objects
        self.num_particles = None
        self.beta=2
        self.alpha=4
        self.support=[]
        dataGen = GeneratorFromImage.GeneratorFromImage(2, 100, cuda=False)
        dataGen.dataConv2d()
        self.stream=TorchStream(dataGen,1000)
        self.radius=10
        self.attach_balls()






#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
