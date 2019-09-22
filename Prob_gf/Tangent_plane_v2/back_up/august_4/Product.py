import sys, pygame
import Quadrants as qu
import Node as nd
import P_trees as tr
import numpy as np
import TangentPlane as tplane


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
        self.dx = 2
        self.L = 1
        self.beta = 2
        self.alpha = 2
        self.Potantial = potential
        self.Interaction = interaction
        self.objects = []
        self.display_size = []
        self.active = False
        self.center=None
        self.std_deviation=None

def potential(x):
    return 50*x**2

def interaction(x, y, alpha):
    return ((x**2 + y**2)**0.5)**(alpha)



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
            m=(p.reg_density+pf.reg_density)/2
            p.metric.append(m)

def update_gradient(status):
    nodes=status.objects
    for node in nodes:
        q=node.objects[0]
        p=q.objects[0]
        p.gradient=[]
        for kid in node.kids:
            qf=kid.objects[0]
            pf=qf.objects[0]
            dE=status.beta/(status.beta-1)*(
                pf.reg_density**(status.beta-1)
                    -p.reg_density**(status.beta-1))
            dE=dE+(potential(qf.shape[0][0])
                -potential(q.shape[0][0]))
            p.gradient.append(dE)

        print(p.gradient)



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
    update_divergence(status)
    update_density(status)
    reg_density(status)
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
                    p.metric[j]*2**status.dx)
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
    self.n=10000
    self.dx=6
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
    a=nd.Node()
    b=qu.Quadrant([[-status.L,status.L]])
    a.objects.append(b)
    qu.Divide(a,status.dx)
    status.objects=tr.Leaves(a)
    for k in range(len(status.objects)):
        anode=status.objects[k]
        anode.objects[0].objects.append(tplane.tangent_plane())
        anode.objects[0].objects[0].num_particles = 0
    i=0
    while i<status.n:
        x=2*status.L
        while not((-status.L<=x) and (x<=status.L)):
            x=np.random.normal(status.center,status.std_deviation)
        b=tr.Find_node([x],a,qu.In)
        p=particle()
        b.objects[0].objects[0].particles.append(p)
        b.objects[0].objects[0].num_particles = b.objects[0].objects[0].num_particles + 1
        p.position.append(b)
        p.velocity.append(b)
        i=i+1
    for k in range(len(status.objects)):
        anode=status.objects[k]
        if k==0:
            anode.kids.append(status.objects[k+1])
        if  (0<k) and (k<len(status.objects)-1):
            anode.kids.append(status.objects[k-1])
            anode.kids.append(status.objects[k+1])
        if k==len(status.objects)-1:
            anode.kids.append(status.objects[k-1])

    #print('Hi')



def plot(status,Display,size=None,tree=None):
    white = 255,255,255
    yellow = 180,180,0
    i=0
    q=tree
    p=tree
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
    ddx=(Lx_f-Lx_o)/(2**status.dx)
    ddy=(Ly_f-Ly_o)/status.n*(2**status.dx)/10
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
