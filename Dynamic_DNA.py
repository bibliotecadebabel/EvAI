from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
from DNA_Phase_space import DNA_Phase_space
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

    def imprimir(self):
        for node in self.objects:
            print('The kid(s) of')
            print(self.node2key(node))
            print('are(is):')
            for nodek in node.kids:
                print(self.node2key(nodek))

    def update_space_default(self):
        pass

    def update_force_field(self):
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
                if p.interaction_field:
                    component=component+p.interaction_field[k]*c_i
                p.force_field.append(component)
                k=k+1

    def update_velocity(self):
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
                            par_dE=force_field**2
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
                    phase_space.mutate(node,node_f)
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




    def __init__(self,space,phase_space,update_space=None):
        self.space = space
        self.phase_space= phase_space
        self.objects=phase_space.objects
        self.support=phase_space.support
        if update_space:
            self.update_space=update_space
        else:
            self.update_space=self.update_space_default
        self.diffusion_coefficient=1
        self.lost_coefficient=1
        self.interaction_coefficient=1
        self.dt=0.01




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
