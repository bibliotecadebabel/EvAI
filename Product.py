import sys, pygame
import Quadrants as qu
import Node as nd
import P_trees as tr
import numpy as np
import TangentPlane as tplane
import Graphs as gr
import math
import V_graphics as cd
import children.Data_generator as dgen
import children.Interfaces as Inter
import children.Operations as Op
import children.net2.Network as nw


# Initialization of parameters
class particle():
    def __init__(self):
        self.position = []
        self.velocity = []
        self.objects=[]

class Status():
    def __init__(self, display_size=None):
        self.dt = 0.01
        self.tau=0.005
        self.n = 10
        self.r=3
        self.dx = 2
        self.L = 1
        self.beta = 2
        self.alpha = 1
        self.Potantial = potential
        self.Interaction = interaction
        self.objects = []
        self.display_size = []
        self.active = False
        self.center=None
        self.std_deviation=None
        self.mouse_frame1=[]
        self.mouse_frame2=[0]
        self.frame1=[]
        self.frame2=[]
        self.Data_gen=None

def potential(x):
    return (x-50)**2

def interaction(r,status):
    return (200*r/(2**status.dx))**(status.alpha)/abs(status.alpha-1)

def update_divergence(status):
    nodes=status.objects
    for node in nodes:
        q=node.objects[0]
        p=q.objects[0]
        p.divergence=0
        for kid in node.kids:
            qf=kid.objects[0]
            pf=qf.objects[0]
            p.divergence=p.divergence+(
                pf.num_particles-p.num_particles)/status.n

def update_density(status):
    nodes=status.objects
    for node in nodes:
        q=node.objects[0]
        p=q.objects[0]
        p.divergence=0
        p.density=p.num_particles/status.n

def update_interaction_field(status):
    nodes=status.objects
    for node in nodes:
        q=node.objects[0]
        p=q.objects[0]
        p.interaction_field=0
        distance=p.distance
        for key in distance:
            node_k=key.objects[0]
            q_k=node_k.objects[0]
            p_k=q_k.objects[0]
            p.interaction_field=(p.interaction_field
                +p_k.density
                    *interaction(distance[key],status))

def reg_density(status):
    nodes=status.objects
    for node in nodes:
        q=node.objects[0]
        p=q.objects[0]
        p.divergence=0
        p.reg_density=p.density+status.tau*p.density

def update_metric(status):
    nodes=status.objects
    for node in nodes:
        q=node.objects[0]
        p=q.objects[0]
        p.metric=[]
        for kid in node.kids:
            qf=kid.objects[0]
            pf=qf.objects[0]
            #m=(p.reg_density+pf.reg_density)/2
            m=1
            p.metric.append(m)

def update_gradient(status):
    #update_interaction_field(status)
    x0=status.mouse_frame2[0]
    x0=0
    nodes=status.objects
    dE=0
    for node in nodes:
        q=node.objects[0]
        p=q.objects[0]
        p.gradient=[]
        for kid in node.kids:
            qf=kid.objects[0]
            pf=qf.objects[0]
            dE=dE+(potential(float(qf.shape)-x0)
                -potential(float(q.shape)-x0))
            dE=dE+100*status.beta*(
                pf.density**(status.beta-1)
                    -p.density**(status.beta-1)
                    /abs(status.beta-1)
                        )
            """dE=dE+(pf.interaction_field
                -p.interaction_field)"""
            p.gradient.append(dE)

        #print(p.gradient)



def update(status):
    update_velocity(status)
    for i in range(len(status.objects)):
        q=status.objects[i].objects[0]
        if q.objects[0].num_particles > 0:
            for particle in q.objects[0].particles:
                if not(particle.position[0]==particle.velocity[0]):
                    a=particle.position[0]
                    b=particle.velocity[0]
                    particle.position=[]
                    particle.velocity=[]
                    q.objects[0].num_particles =q.objects[0].num_particles - 1
                    particle.position.append(b)
                    particle.velocity.append(b)
                    qf=b.objects[0]
                    qf.objects[0].num_particles=qf.objects[0].num_particles+1
                    q.objects[0].particles.remove(particle)
                    qf.objects[0].particles.append(particle)
                    ##print(q.objects[0].num_particles," - ", qf.objects[0].num_particles, " || LONGITUD REAL ||  ", len(q.objects[0].particles)," - ", len(qf.objects[0].particles))

def update_velocity(status):
    #update_divergence(status)
    update_density(status)
    #reg_density(status)
    update_metric(status)
    update_gradient(status)
    l = len(status.objects)
    for i in range(l):
        #print(i)
        node=status.objects[i]
        q=status.objects[i].objects[0]
        p=q.objects[0]
        dE=0
        for j in range(len(node.kids)):
            if p.gradient[j]<0:
                dE=dE+(p.gradient[j]**2)*(
                    p.metric[j])
        dE=dE**(0.5)
        for particle in q.objects[0].particles:
            prob = np.random.uniform(0,1)
            if prob < status.dt*abs(dE):
                j=0
                par_dE=0
                if p.gradient[j]<=0:
                    par_dE=(p.gradient[j]**2)*(
                        p.metric[j])
                while par_dE<prob**2 and j+1<len(node.kids):
                    j=j+1
                    if p.gradient[j]<=0:
                        par_dE=(p.gradient[j]**2)*(
                            p.metric[j])
                particle.position=[]
                particle.position.append(status.objects[i])
                particle.velocity=[]
                particle.velocity.append(
                    particle.position[0].kids[j])
            else:
                pass




def initialize_parameters(self):
    display_size=[1000,500]
    self.dt=0.1
    self.n=100
    self.dx=100
    self.L=1
    self.beta=2
    self.alpha=2
    self.center=.5
    self.std_deviation=1
    self.Potantial=potential
    self.Interaction=interaction
    self.display_size=display_size

#Here the objects of status are a list of nodes
#The objects of nodes are quadrants which have the physical Range
# the objects of quadrants are:
# in position 0 a list of particles
# in position 1 the size of such list

def create_objects(status):
    status.Data_gen=dgen.Data_gen()
    status.Data_gen.gen_data()
    def add_node(g,i):
            node=nd.Node()
            q=qu.Quadrant(i)
            p=tplane.tangent_plane()
            node.objects.append(q)
            q.objects.append(p)
            g.add_node(i,node)
            #status.objects.append(node)
    #Initializes graph
    g=gr.Graph()
    add_node(g,0)
    k=0
    while k<status.dx:
        add_node(g,k+1)
        g.add_edges(k,[k+1])
        k=k+1
    k=0
    status.objects=list(g.key2node.values())
    node=status.objects[0]
    p=node_plane(node)
    #Initializes particles
    while k<status.n:
        par=particle()
        par.position.append(node)
        par.velocity.append(node)
        #print(status.Data_gen.size)
        par.objects.append(nw.Network([status.Data_gen.size[0],
            status.Data_gen.size[1],2]))
        p.particles.append(par)
        p.num_particles+=1
        k=k+1
    #Initializes conectivity radius
    attach_balls(status,status.r)

def node_shape(node):
    q=node.objects[0]
    return q.shape

def node_plane(node):
    q=node.objects[0]
    return q.objects[0]
#this process atacches at each node of status.objects a spaning P_tree
#for the graph centered at the given node and build with r-degrees of conectivity
#it also attaches the disctionary that to each node in the spanning_tree
#assings its distance to the parent node

def attach_balls(status,n_r):
    for node in status.objects:
        q=node.objects[0]
        p=q.objects[0]
        p.ball=gr.spanning_tree(node,n=n_r)
        p.distance=tr.tree_distances(p.ball)

            #print('Hi')



def plot(status,Display,size=None,tree=None):
    white = 255,255,255
    yellow = 180,180,0
    i=0
    q=tree
    p=tree
    width=p.objects[0].shape[0][1]
    Height=q.objects[0].shape[1][1]
    scale=size[2]
    while i<scale:
        q=q.kids[0]
        i=i+1
    lx=q.objects[0].shape[0][1]-q.objects[0].shape[0][0]
    ly=q.objects[0].shape[1][1]-q.objects[0].shape[1][0]
    Lx_o=lx*size[0][0]
    Lx_f=lx*size[0][1]
    Ly_o=p.objects[0].shape[1][1]-ly*size[1][0]
    Ly_f=p.objects[0].shape[1][1]-ly*size[1][1]
    ddx=(Lx_f-Lx_o)/(status.dx)
    ddy=(Ly_f-Ly_o)/status.n*2
    status.frame1=[[0,Height],[1,0],[0,-1]]
    status.frame2=[[Lx_o+(Lx_f-Lx_o)/2,Ly_o/2*4/5],[(Lx_f-Lx_o)/2,0],[0,Ly_f-Ly_o/2*4/5]]
    frame1=status.frame1
    frame2=status.frame2
    mouse=[status.mouse_frame1[0],status.mouse_frame1[1]]
    status.mouse_frame2=cd.coord2vector(mouse,frame1,frame2)
    print(len(status.objects))
    #print(status.mouse_frame2)
#    positions=[[300,300],[400,400]]
    #pygame.draw.aaline(Display,white,positions[0],positions[1],True)
    for i in range(len(status.objects)-1):
        px_o=Lx_o+i*ddx
        px_f=Lx_o+(i+1)*ddx
        py_o=Ly_o+status.objects[i].objects[0].objects[0].num_particles*ddy
        py_f=Ly_o+status.objects[i+1].objects[0].objects[0].num_particles*ddy
        positions=[[px_o,py_o],[px_f,py_f]]
#        print(positions)
        pygame.draw.aaline(Display,white,positions[0],positions[1],True)

status=Status()
initialize_parameters(status)
create_objects(status)
print('Done')
p=node_plane(status.objects[2])
print(p.num_particles)


"""
c=[]
d=[]
b=[5]
c.append(b)
d.append(b)
b[0]=3
print(c)
print(d)
d[0]
#Program execution"""

"""status=Status()
initialize_parameters(status)
create_objects(status)"""
