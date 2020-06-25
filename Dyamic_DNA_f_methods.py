from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
from DNA_Phase_space import DNA_Phase_space
from timing import timing
import numpy as np
import DNA_graph_functions as Funct
from random import randint
from Particle import particle as particle


def update_force_field_ac(self):
    c_k=self.mutation_coefficient
    for node in self.support:
        p=self.node2plane(node)
        p.force_field=[]
        for kid in node.kids:
            p_k=self.node2plane(kid)
            d_phi=p_k.velocity_potential-p.velocity_potential
            component=0
            if d_phi>0:
                component=d_phi
            p.force_field.append(-c_k*component)




def update_velocity_mobility(self):
    dt=self.dt
    phase_space=self.phase_space
    for node in self.support:
        p=self.node2plane(node)
        dE=0
        force_field=p.force_field
        dE=sum([force for force in force_field if force<0])
        for particle in p.particles:
            prob = np.random.uniform(0,1)
            variance=phase_space.node2variance(node)
            if variance:
                if int(variance)>0:
                    mobility=1/variance^5
                else:
                    mobility=1
            else:
                mobility=1
            if prob < dt*abs(dE)*mobility:
                j=0
                par_dE=0
                if force_field[j]<=0:
                    par_dE=abs(force_field[j])
                while par_dE*mobility*dt<prob and j+1<len(node.kids):
                    j=j+1
                    if force_field[j]<=0:
                        par_dE+=abs(force_field[j])
                particle.position=[]
                particle.position.append(node)
                particle.velocity=[]
                particle.velocity.append(node.kids[j])
            else:
                pass


def update_velocity_markov(self):
    dt=self.dt
    for node in self.support:
        p=self.node2plane(node)
        dE=0
        force_field=p.force_field
        dE=sum([force for force in force_field if force<0])
        for particle in p.particles:
            prob = np.random.uniform(0,1)
            if prob < dt*abs(dE):
                j=0
                par_dE=0
                if force_field[j]<=0:
                    par_dE=abs(force_field[j])
                while par_dE*dt<prob and j+1<len(node.kids):
                    j=j+1
                    if force_field[j]<=0:
                        par_dE+=abs(force_field[j])
                particle.position=[]
                particle.position.append(node)
                particle.velocity=[]
                particle.velocity.append(node.kids[j])
            else:
                pass

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

def update_none(self):
    pass


def update_dynamic(self):
    phase_space=self.phase_space
    creator=self.Creator
    selector=self.Selector
    version=self.version
    phase_space.time=phase_space.time+1
    if phase_space.time>20:
        phase_space.time=0
        node2remove=phase_space2node2remove(phase_space)
        if node2remove:
            remove_node(phase_space,node2remove)
            while len(phase_space.objects)<8:
                new_DNA=add_node(phase_space,selector)



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

def add_node(phase_space,Selector,particles=0):
    center=phase_space.center()
    DNA_graph=phase_space.DNA_graph
    graph=DNA_graph.graph
    keys=list(graph.node2key.values())
    new_DNA=None
    while (new_DNA in keys) or new_DNA==None:
        new_DNA=Selector.DNA2new_DNA(center)
    actions=Selector.get_predicted_actions()
    direction=tuple(actions[0])
    Funct.add_node(graph,new_DNA)
    graph.add_edges(center,[new_DNA])
    new_node=phase_space.key2node(new_DNA)
    DNA_graph.objects.append(new_node)
    p=Funct.node2plane(new_node)
    p.direction=direction

    return new_DNA


def remove_node(phase_space,node2remove):
    DNA2remove=phase_space.node2key(node2remove)
    phase_space.objects.remove(node2remove)
    if node2remove in phase_space.support:
        phase_space.support.remove(node2remove)
    DNA_graph=phase_space.DNA_graph
    graph=DNA_graph.graph
    graph.remove_node(DNA2remove)

def node_max_particles(phase_space):
    particles=[
        phase_space.node2particles(node) for node in
            phase_space.objects
            ]
    max_index=np.argmax(np.array(particles))
    return phase_space.objects[max_index]

def phase_space2node2remove(phase_space):
    nodes=phase_space.objects
    support_complement=[ node for node in nodes if
        phase_space.node2particles(node) == 0]
    if support_complement:
        energies=[
            phase_space.node2energy(node) for node in
                support_complement
                ]
        index2remove=np.argmin(np.array(energies))
        node2remove=support_complement[index2remove]
        return node2remove
    else:
        pass
        """
        particles=[
            phase_space.node2particles(node) for node in
                phase_space.support
                ]
        index2remove=np.argmin(np.array(particles))
        node2remove=phase_space.support[index2remove]
        return node2remove"""



def update_from_select_stochastic(self):
    phase_space=self.phase_space
    creator=self.Creator
    selector=self.Selector
    version=self.version
    phase_space.time=phase_space.time+1
    p=randint(0,300)
    if phase_space.node_max_particles:
        node_max = phase_space.node_max_particles
        node_c = phase_space.key2node(phase_space.DNA_graph.center)
        p_c=Funct.node2num_particles(node_c)
        p_m=Funct.node2num_particles(node_max)
    if (phase_space.max_changed and p_m*0.9>p_c) or (
        phase_space.time>4000):
        phase_space.time=0
        num_particles = phase_space.num_particles
        old_graph = phase_space.DNA_graph
        old_center= old_graph.center
        condition = old_graph.condition
        typos = old_graph.typos
        node_max = phase_space.node_max_particles
        center = phase_space.node2key(node_max)
        selector.update(old_graph,new_center=center)
        actions=selector.get_predicted_actions()
        x = old_graph.x_dim
        y = old_graph.y_dim
        space=DNA_Graph(center,1,(x,y),condition,actions,
            version,creator)
        phase_space.DNA_graph = space
        phase_space.objects = space.objects
        phase_space.support=[]
        phase_space.create_particles(num_particles+1)
        phase_space.stream.signals_off()
        phase_space.attach_balls()
        phase_space.max_changed = False
        phase_space.node_max_particles = None
        self.space = space
        self.phase_space= phase_space
        self.objects=phase_space.objects
        self.support=phase_space.support
        self.Graph=phase_space.DNA_graph


def update_from_select_Alai(self):
    phase_space=self.phase_space
    creator=self.Creator
    selector=self.Selector
    version=self.version
    phase_space.time=phase_space.time+1
    node_max=node_max_particles(phase_space)
    node_c = phase_space.key2node(phase_space.DNA_graph.center)
    p_c=Funct.node2num_particles(node_c)
    p_m=Funct.node2num_particles(node_max)
    print(f'The value of p_m is : {p_m} and p_c is : {p_c} ')
    if (p_m>p_c*2) or (
        phase_space.time>4000):
        phase_space.time=0
        num_particles = phase_space.num_particles
        old_graph = phase_space.DNA_graph
        old_center= old_graph.center
        condition = old_graph.condition
        typos = old_graph.typos
        node_max = phase_space.node_max_particles
        center = phase_space.node2key(node_max)
        selector.update(old_graph,new_center=center)
        actions=selector.get_predicted_actions()
        x = old_graph.x_dim
        y = old_graph.y_dim
        space=DNA_Graph(center,1,(x,y),condition,actions,
            version,creator)
        phase_space.DNA_graph = space
        phase_space.objects = space.objects
        phase_space.support=[]
        phase_space.create_particles(num_particles+1)
        phase_space.stream.signals_off()
        phase_space.attach_balls()
        phase_space.max_changed = False
        phase_space.node_max_particles = None
        self.space = space
        self.phase_space= phase_space
        self.objects=phase_space.objects
        self.support=phase_space.support
        self.Graph=phase_space.DNA_graph
    elif False:
    #elif phase_space.time>self.clear_period:
        phase_space.time=0
        node2remove=phase_space2node2remove(phase_space)
        node_c = phase_space.key2node(phase_space.DNA_graph.center)
        if node2remove and not (node2remove==node_c):
            remove_node(phase_space,node2remove)
            new_DNA=add_node(phase_space,selector)

def update_from_select_09(self):
    phase_space=self.phase_space
    creator=self.Creator
    selector=self.Selector
    version=self.version
    phase_space.time=phase_space.time+1
    node_max=node_max_particles(phase_space)
    node_c = phase_space.key2node(phase_space.DNA_graph.center)
    p_c=Funct.node2num_particles(node_c)
    p_m=Funct.node2num_particles(node_max)
    print(f'The value of p_m is : {p_m} and p_c is : {p_c} ')
    if (p_m>p_c*2):
        phase_space.time=0
        num_particles = phase_space.num_particles
        old_graph = phase_space.DNA_graph
        old_center= old_graph.center
        condition = old_graph.condition
        typos = old_graph.typos
        #node_max = phase_space.node_max_particles
        node_max = node_max_particles(phase_space)
        print("node_max particles:", Funct.node2num_particles(node_max))
        time.sleep(5)
        center = phase_space.node2key(node_max)
        status=phase_space.status
        Alai=status.Alai
        stream=phase_space.stream
        delta=stream.key2len_hist(center)
        Alai.update(delta)
        stream.signals_off()
        stream.key2signal_on(center)
        stream.clear()
        selector.update(old_graph,new_center=center)
        actions=selector.get_predicted_actions()
        x = old_graph.x_dim
        y = old_graph.y_dim
        space=DNA_Graph(center,1,(x,y),condition,actions,
            version,creator)
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
    #elif False:
    #elif phase_space.time>self.clear_period:
    #    phase_space.time=0
    #    node2remove=phase_space2node2remove(phase_space)
    #    node_c = phase_space.key2node(phase_space.DNA_graph.center)
    #    if node2remove and not (node2remove==node_c):
    #        remove_node(phase_space,node2remove)
    #        new_DNA=add_node(phase_space,selector)


def update_from_select(self):
    phase_space=self.phase_space
    creator=self.Creator
    selector=self.Selector
    version=self.version
    phase_space.time=phase_space.time+1
    if phase_space.max_changed or phase_space.time>2000:
        phase_space.time=0
        num_particles = phase_space.num_particles
        old_graph = phase_space.DNA_graph
        old_center= old_graph.center
        condition = old_graph.condition
        typos = old_graph.typos
        node_max = phase_space.node_max_particles
        center = phase_space.node2key(node_max)
        selector.update(old_graph,new_center=center)
        actions=selector.get_predicted_actions()
        x = old_graph.x_dim
        y = old_graph.y_dim
        space=DNA_Graph(center,1,(x,y),condition,actions,
            version,creator)
        phase_space.DNA_graph = space
        phase_space.objects = space.objects
        phase_space.support=[]
        phase_space.create_particles(num_particles+1)
        phase_space.stream.signals_off()
        phase_space.attach_balls()
        phase_space.max_changed = False
        phase_space.node_max_particles = None
        self.space = space
        self.phase_space= phase_space
        self.objects=phase_space.objects
        self.support=phase_space.support
        self.Graph=phase_space.DNA_graph


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
