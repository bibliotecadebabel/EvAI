from Geometric.Particle import particle as particle
from utilities.Abstract_classes.classes.torch_stream_fb import TorchStream
from DAO import GeneratorFromImage
import children.pytorch.MutateNetwork_Dendrites_clone as Mutator
import utilities.Graphs as gr
import utilities.P_trees as tr
from Geometric.timing import timing
import mutations.mutation_manager as mutation_manager

class DNA_Phase_space():

    def node2velocity_potential(self,node):
        p=self.node2plane(node)
        return p.velocity_potential

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

    def node2direction(self,node):
        p=self.node2plane(node)
        return p.direction

    def node2particles(self,node):
        p=self.node2plane(node)
        return p.num_particles

    def node2variance(self,node):
        p=self.node2plane(node)
        return p.variance

    def add_net(self,key):
        stream=self.stream
        stream.add_net(key)
        pass

    def node2energy(self,node):
        p=self.node2plane(node)
        return p.energy

    def node2V(self,node):
        stream=self.stream
        k_o=self.node2key(node)
        V_o=stream.findCurrentvalue(k_o)
        p=self.node2plane(node)
        if V_o:
            #p.energy=V_o
            p.energy=stream.key2average(k_o)
            """
            p.energy=(V_o/(0.01+V_o)
            +V_o/(0.001+V_o)
            +V_o/(0.0001+V_o)
            +V_o/(0.00001+V_o)
            +V_o/(0.00001+V_o)
            +V_o/.001+(V_o/.1)**2)
            """
            return p.energy
        else:
            p.energy=V_o
            return V_o

    def update_max_particles(self):
        old_max=self.node_max_particles
        if self.support:
            node_M=self.support[0]
            max=self.node2particles(node_M)
            for node in self.support:
                if  self.node2particles(node)>max:
                    node_M=node
            if old_max and not(old_max == node_M):
                self.max_changed = True
            self.node_max_particles=node_M
            self.DNA_graph.node_max_particle=node_M

        else:
            self.node_max_particles=None
            self.DNA_graph.node_max_particle=None

    def print_max_particles(self):
        if self.node_max_particles:
            node=self.node_max_particles
            key=self.node2key(node)
            plane=self.node2plane(node)
            print('The particles of {} are {}'.format(key, plane.num_particles))
            print('and its kid(s) ha(ve)(s):')
            for nodek in node.kids:
                key=self.node2key(nodek)
                plane=self.node2plane(nodek)
                if plane.num_particles>0:
                    print('The particles of {} are {}'.format(key, plane.num_particles))


    def node2net(self,node):
        stream=self.stream
        return stream.get_net(self.node2key(node))

    def mutate(self,node_o,node_f):
        k_f=self.node2key(node_f)
        stream=self.stream

        if not(stream.get_net(k_f)):

            net=self.node2net(node_o)
            path_mutations = node_f.objects[0].objects[0].path
            net_f = None

            old_network = net
            for kid_dna in path_mutations:
                net_f = self.Mutator.execute_mutation(old_network, kid_dna)
                stream.memoryManager.deleteNetwork(old_network)
                old_network = net_f

            stream.add_node(k_f)
            stream.link_node(k_f,net_f)
            stream.charge_node(k_f)

    def interction_kernel(self,r):
        alpha=self.alpha
        return  -1*(r/self.influence)**(alpha)

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
        #self.add_net(key)
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
                    if not(V_f):
                        V_f=V_o
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

    def emptynumber(self,number):
        if number:
            return number
        else:
            return 100

    def update_variance(self):
        for node in self.support:
            variance=sum([self.node2V(kid)**2 for kid
                in node.kids if self.emptynumber(self.node2energy(kid))<self.node2energy(node)])
            self.variance=variance**(0.5)


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

    def print_balls(self):
        for node in self.objects:
            key = self.node2key(node)
            p = self.node2plane(node)
            distance=p.distance
            print('The distances from' )
            print(key)
            print('are')
            for key in distance.keys():
                print('To node')
                print(self.node2key(key.objects[0]))
                print('is')
                print(distance.get(key))

    def print_support(self):
        for node in self.support:
            key = self.node2key(node)
            print(f'{self.node2direction(node)} : { self.node2particles(node)} : {self.node2velocity_potential(node)}: {self.node2energy(node)}')

    def print_center(self):
        node = self.key2node(self.center())
        stream=self.stream
        k_o=self.node2key(node)
        V_o=stream.findCurrentvalue(k_o)
        if node:
            print(f'{self.center()} : {self.node2particles(self.key2node(self.center()))} and its energy is {V_o}')

    def computeNesterov(self, node_parent):

        tangent_plane_parent = node_parent.objects[0].objects[0]
        nesterov_value = 0

        for node_kid in node_parent.kids:
            tangent_plane_kid = node_kid.objects[0].objects[0]
            value = (tangent_plane_kid.energy - tangent_plane_parent.energy)*(tangent_plane_kid.velocity_potential - tangent_plane_parent.velocity_potential)*tangent_plane_kid.num_particles
            nesterov_value += value
        
        return nesterov_value

    def updateNesterov(self):

        node_center = self.key2node(self.DNA_graph.center)       
        total_nesterov = 0

        for node in self.objects:
            total_nesterov += self.computeNesterov(node)

        print("total nesterov value = ", total_nesterov)

        if total_nesterov >= 0 or len(self.support) == 1:

            node_center.objects[0].objects[0].velocity_potential = 0

            for node_kid in node_center.kids:
                node_kid.objects[0].objects[0].velocity_potential = 0

            for node in self.objects:
                print("velcity potential node: ", node.objects[0].objects[0].velocity_potential)
            
    def update(self):
        print('The center is')
        self.print_center()
        print('The support is')
        self.print_support()
        stream=self.stream
        #print('The signals are')
        #stream.print_signal()
        print('Charging took:')
        timing(stream.charge_nodes)
        stream.signals_off()
        self.DNA_graph.update()
        self.update_density()
        self.update_diffussion_field()
        print('Computing the external field took:')
        timing(self.update_external_field)

        enable_nesterov = True
        if enable_nesterov == True:
            self.updateNesterov()
        #print('After external field, the signals are')
        #stream.print_signal()
        #print('Computing the interaction field took:')
        #timing(self.update_interaction_field)
        #print('Computing maximum took:')
        self.update_max_particles()
        self.update_variance()
        #print('The maximum node is')
        #print(self.node2key(self.node_max_particles))
        stream.pop()
        #print('After pop, the signals are')





    #It seems the current version cannot handle regularization and negarive
    #Pourus medium exponent

    def __init__(self,DNA_graph,
            Potential=None,Interaction=None,External=None,
            stream=None,status=None):
        self.DNA_graph = DNA_graph
        self.objects = DNA_graph.objects
        self.num_particles = None
        self.beta=2
        self.alpha=50
        self.support=[]
        self.stream=stream
        self.radius=10
        self.influence=2
        self.node_max_particles=None
        self.max_changed=False
        self.attach_balls()
        self.time=0
        if status:
            mut_man=mutation_manager.MutationManager(status.version)
            self.Mutator=mut_man.get_mutation_function()
            self.status=status
        else:
            self.Mutator=Mutator






#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
