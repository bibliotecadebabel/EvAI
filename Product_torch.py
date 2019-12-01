import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Safe as sf
import utilities.P_trees as tr
import numpy as np
import TangentPlane as tplane
import utilities.Graphs as gr
import math
import V_graphics as cd
import Transfer.Transfer as tran
import children.Data_generator as dgen
import children.Interfaces as Inter
import children.Operations as Op
import children.net2.Network as nw
from DAO import GeneratorFromImage
from utilities.Abstract_classes.classes.torch_stream import TorchStream
import children.pytorch.Network as nw
import time


# Initialization of parameters
class particle():
    def __init__(self):
        self.position = []
        self.velocity = []
        self.objects=[]

class Status():
    def __init__(self, display_size=None):
        self.dt = 0.1
        self.tau=0.01
        self.n = 1000
        self.r=3
        self.dx = 20
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
        self.Transfer=None
        self.S=100
        self.Comp=2
        self.Data_gen=None
        self.p=1
        self.display=None
        self.scale=None
        self.sectors=None
        self.nets={}
        self.stream=None
        self.Graph=None

def print_nets(status):
    if status is None:
        print('IMPOSIBLE')
    for node in status.objects:
        key=node_shape(node)
        net=status.nets.get(key)
        if not(net==None):
            print('The energy of ',key,' is:',net.total_value)



def update_nets(status):
    for node in status.objects:
        node_energy(node,status)
        p=node_plane(node)
        particles=p.particles
        key=node_shape(node)
        if not(p.num_particles==0):
            par=particles[0]
            net=par.objects[0]
            net.Training(status.Data_gen.Data,dt=status.tau,p=2)
        #if (key in status.nets.keys()):
        #    net=status.nets.get(key)
        #    if not(net==None):
        #        net.Training(status.Data_gen.Data,dt=status.tau,p=2)
    return

def potential(x,status=None):
    return node_energy(status.objects[x],status)

def r_potential(x):
    return -1/x

def d_potential(b,a,status=None):
    plane_a=node_plane(status.objects[a])
    plane_b=node_plane(status.objects[b])
    a_key=a
    b_key=b
    u=0
    if not(b_key in status.nets.keys()) and plane_a.num_particles!=0:
        par=plane_a.particles[0]
        net=par.objects[0]
        net_clone=net.clone()
        if b_key>a_key:
            net_clone.addFilters()
        elif a_key>b_key:
            net_clone.deleteFilters()
        else:
            print('ERROR')
        net_clone.Training(status.Data_gen.Data,dt=status.tau,p=2)
        plane_a=node_plane(status.objects[a])
        u=(r_potential(net_clone.total_value)
            -r_potential(potential(a_key,status)))
    else:
        if plane_a.num_particles==0:
            u=0
        else:
            u=(r_potential(potential(b_key,status))
                -r_potential(potential(a_key,status)))
    if not(u==0):
        return(1/u)
    else:
        return(u)



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
    nodes=status.objects
    for node in nodes:
        q=node.objects[0]
        p=q.objects[0]
        p.gradient=[]
        for kid in node.kids:
            dE=0
            qf=kid.objects[0]
            pf=qf.objects[0]
            dE=dE+(d_potential(int(qf.shape),int(q.shape),status))
            if dE==0:
                dE=dE+1*status.beta*(
                    pf.density**(status.beta-1)
                        -p.density**(status.beta-1)
                        )
            else:
                dE=dE+(1+abs(dE))*status.beta*(
                    pf.density**(status.beta-1)
                        -p.density**(status.beta-1)
                        )
            """dE=dE+(pf.interaction_field
                -p.interaction_field)"""
            p.gradient.append(dE)

        #print(p.gradient)



def update(status):
    update_nets(status)
    #time.sleep(10)
    status.Transfer.status=status
    status.Transfer.update()
    update_velocity(status)
    for i in range(len(status.objects)):
        q=status.objects[i].objects[0]
        if q.objects[0].num_particles > 0:
            for particle in q.objects[0].particles:
                #print(q.shape)
                if not(particle.position[0]==particle.velocity[0]):
                    a=particle.position[0]
                    b=particle.velocity[0]
                    net=particle.objects[0]
                    a_key=node_shape(a)
                    b_key=node_shape(b)
                    b_plane=node_plane(b)
                    if not(b_key in status.nets.keys()):
                        net_b=net.clone()
                        if b_key>a_key:
                            net_b.addFilters()
                        elif a_key>b_key:
                            net_b.deleteFilters()
                        sf.safe_update(status.nets,b_key,net_b)
                    particle.objects[0]=status.nets[b_key]
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
    self.dt=0.01
    self.n=1000
    self.dx=20
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
    status.Data_gen=GeneratorFromImage.GeneratorFromImage(
    status.Comp, status.S, cuda=False)
    status.Data_gen.dataConv2d()
    def create_DNA(filters):
        ks=[filters+2]
        dataGen=status.Data_gen
        x = dataGen.size[1]
        y = dataGen.size[2]
        networkADN = ((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
        return networkADN
    def add_node(g,i):
        DNA=create_DNA(i)
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)
            #status.objects.append(node)
    #Initializes graph
    g=gr.Graph()
    add_node(g,0)
    k=0
    while k<status.dx:
        add_node(g,k+1)
        g.add_edges(create_DNA(k),[create_DNA(k+1)])
        k=k+1
    k=0
    status.objects=list(g.key2node.values())
    status.Graph=g

    DNA=create_DNA(0)
    status.stream=TorchStream(status.Data_gen,100)
    network = nw.Network(DNA,
                         cudaFlag=False)
    status.stream.add_node(DNA)
    status.stream.link_node(DNA,network)
    log=status.stream.key2log(DNA)

    node=status.Graph.key2node[DNA]
    p=node_plane(node)
    #Initializes particles
    while k<status.n:
        par=particle()
        par.position.append(node)
        par.velocity.append(node)
        #print(status.Data_gen.size)
        par.objects.append(log)
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

def node_energy(node,status=None):
    p=node_plane(node)
    if p.num_particles==0:
        key=node_shape(node)
        if not(key in status.nets.keys()):
            p.energy=None
        else:
            net=status.nets[key]
            p.energy=net.total_value
    else:
        par=p.particles[0]
        net=par.objects[0]
        p.energy=net.total_value
    return p.energy


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
    #mouse=[status.mouse_frame1[0],status.mouse_frame1[1]]
    #status.mouse_frame2=cd.coord2vector(mouse,frame1,frame2)
    #print(len(status.objects))
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
create_objects(status)
print('Done')

"""
status=Status()
initialize_parameters(status)
create_objects(status)
status.Transfer=tran.TransferRemote(status,
    'remote2local.txt','local2remote.txt')
status.Transfer.un_load()
status.Transfer.write()
k=0
while False:
    update(status)
    status.Transfer.un_load()
    status.Transfer.write()
    transfer=status.Transfer.status_transfer
    k=k+1
    pass
while True:
    status.Transfer.readLoad()
    if status.active:
        update(status)
        print_nets(status)
#        time.sleep(0.5)
    else:
        print('inactive')
    k=k+1
"""


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
