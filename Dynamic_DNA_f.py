from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
from DNA_Phase_space import DNA_Phase_space
from timing import timing
import numpy as np

class Dynamic_DNA():

    def node2plane(self,node):
        q=node.objects[0]
        return q.objects[0]

    def key2node(self,key):
        return self.graph.key2node[key]

    def node2key(self,node):
        q=node.objects[0]
        return q.shape

    def update_space_default(self):
        phase_space=self.phase_space
        if phase_space.max_changed:
            num_particles = phase_space.num_particles
            old_graph = phase_space.DNA_graph
            condition = old_graph.condition
            typos = old_graph.typos
            node_max = phase_space.node_max_particles
            center = phase_space.node2key(node_max)
            x = old_graph.x_dim
            y = old_graph.y_dim
            space=DNA_Graph(center,self.dx,(x,y),condition,typos,'inclusion')
            #Phase_space = DNA_Phase_space(space)
            phase_space.DNA_graph = space
            phase_space.objects = space.objects
            phase_space.support=[]
            phase_space.create_particles(num_particles+1)
            phase_space.attach_balls()
            phase_space.max_changed = False
            phase_space.node_max_particles = None
            self.space = space
            self.phase_space= phase_space
            self.objects=phase_space.objects
            self.support=phase_space.support
            self.Graph=phase_space.DNA_graph

    def update_force_field_default(self):
        c_k=self.mutation_coefficient
        for node in self.support:
            p=self.node2plane(node)
            p.force_field=[]
            k=0
            c_d=self.diffusion_coefficient
            c_l=self.lost_coefficient
            c_i=self.interaction_coefficient
            for kid in node.kids:
                component=0
                if p.difussion_field:
                    component=component+p.difussion_field[k]*c_d
                if p.external_field:
                    component=component+p.external_field[k]*c_l
                    #-p.external_field[k]/abs(p.external_field[k]+self.epsilon)*c_l
                if p.interaction_field:
                    component=component+p.interaction_field[k]*c_i
                p.force_field.append(c_k*component)
                k=k+1

    def update_velocity_potential(self):
        c_p=self.momentum
        for node in self.objects:
            p=self.node2plane(node)
            variance=p.variance
            dt=self.dt
            p.velocity_potential+=-1/c_p*dt*p.energy
            if p.variance:
                for kid in node.kids:
                    p_k=self.node2plane(kid)
                    d_phi=p_k.velocity_potential-p.velocity_potential
                    if d_phi>0:
                        p.velocity_potential+=-dt*d_phi**2/(
                            4*len(node.kids)*(p.variance+0.1)**(-5))


    def update_velocity_default(self):
        dt=self.dt
        for node in self.support:
            p=self.node2plane(node)
            dE=0
            force_field=p.force_field
            for j in range(len(node.kids)):
                if force_field[j]<0:
                    dE=dE+(force_field[j]**2)
            dE=dE**(0.5)
            for particle in p.particles:
                prob = np.random.uniform(0,1)
                if prob < dt*abs(dE):
                    j=0
                    par_dE=0
                    if force_field[j]<=0:
                        par_dE=(force_field[j]**2)
                    while par_dE<prob**2 and j+1<len(node.kids):
                        j=j+1
                        if force_field[j]<=0:
                            par_dE=force_field[j]**2
                    particle.position=[]
                    particle.position.append(node)
                    particle.velocity=[]
                    particle.velocity.append(node.kids[j])
                else:
                    pass

    def update_particles(self):
        phase_space=self.phase_space
        for node in self.support:
            p=self.node2plane(node)
            for particle in p.particles:
                node_f=particle.velocity[0]
                if not(node==node_f):
                    plane_f=self.node2plane(node_f)
                    #phase_space.mutate(node,node_f)
                    particle.position=[]
                    particle.velocity=[]
                    particle.position.append(node_f)
                    particle.velocity.append(node_f)
                    if plane_f.num_particles==0:
                        self.support.append(node_f)
                    p.num_particles=p.num_particles-1
                    plane_f.num_particles=plane_f.num_particles+1
                    p.particles.remove(particle)
                    plane_f.particles.append(particle)
                    if p.num_particles==0:
                        self.support.remove(node)

    def print_particles(self):
        Graph=self.Graph
        dict=Graph.graph.key2node
        for item in dict.items():
            key=item[0]
            node=item[1]
            space=self.phase_space
            plane=space.node2plane(node)
            log=node.get_object()
            print('The particles of {} are {}'.format(key, plane.num_particles))


    def update(self):
        self.phase_space.update()
        if not(self.update_force_field==self.
            update_force_field_default):
            self.update_velocity_potential()
            self.update_force_field(self)
        else:
            self.update_force_field()
        self.update_velocity(self)
        self.update_particles()
        self.update_space(self)







    def __init__(self,space,phase_space,dx=1,
        update_space=None,
        update_velocity=None,
        update_force_field=None,
        Selector=None,Creator=None,version='inclusion',
        mutation_coefficient=.1,
        clear_period=80,
        Alai=None):
        self.Alai=Alai
        self.space = space
        self.phase_space= phase_space
        self.objects=phase_space.objects
        self.support=phase_space.support
        if update_space:
            self.update_space=update_space
        else:
            self.update_space=self.update_space_default
        if update_velocity:
            self.update_velocity=update_velocity
        else:
            self.update_velocity=self.update_velocity_default
        if update_force_field:
            self.update_force_field=update_force_field
        else:
            self.update_force_field=self.update_force_field_default
        self.diffusion_coefficient=0
        self.lost_coefficient=50
        self.interaction_coefficient=0
        self.dt=0.01
        self.mutation_coefficient=mutation_coefficient
        self.clear_period=clear_period
        self.Graph=phase_space.DNA_graph
        self.epsilon=0.001
        self.dx=dx
        self.Selector=Selector
        self.Creator=Creator
        self.version=version
        self.momentum=2




#        creator=Directions.get(type)
#        g=creator(self)
#        self.graph=g
#        self.objects=list(g.key2node.values())






#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
