import sys, pygame
import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.P_trees as tr
import numpy as np
import TangentPlane as tplane
import utilities.Graphs as gr
import math
import V_graphics as cd
import Transfer.Transfer as tran

#import children.Interfaces as Inter
#import children.Operations as Op
#import children.net2.Network as nw
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
        self.dx = 8
        self.L = 1
        self.beta = 2
        self.alpha = 50
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
#        self.Data_gen=None
        self.p=1
        self.display=None
        self.scale=None
        self.sectors=None
        self.typos=(0,(0,0,1,1))
    def print_particles(self):
        k=0
        for node in self.objects:
            p=node2plane(node)
            print('The particels in')
            print(k)
            print('are')
            print(p.num_particles)
            k=k+1


def node2quadrant(node):
    return node.get_object()

def node2plane(node):
    quadrant=node2quadrant(node)
    return quadrant.objects[0]

def potential(x):
    return (x-50)**2

        #print(p.gradient)



def update(status):
    status.Transfer.status=status
    status.Transfer.update()

def initialize_parameters(self):
    display_size=[1000,500]
    self.dt=0.01
    self.n=1000
    self.dx=8
    self.L=1
    self.beta=2
    self.alpha=50
    self.center=.5
    self.std_deviation=1
    self.display_size=display_size
    self.typos=(0,(0,0,1,1))

#Here the objects of status are a list of nodes
#The objects of nodes are quadrants which have the physical Range
# the objects of quadrants are:
# in position 0 a list of particles
# in position 1 the size of such list

def create_objects(status):
    status.Transfer=tran.TransferLocal(status,
        'local2remote.txt','remote2local.txt')
    status.Transfer.un_load()
    status.Transfer.write()
    #status.Data_gen=dgen.Data_gen()
    #status.Data_gen.gen_data()
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
    #    par.objects.append(nw.Network([status.Data_gen.size[0],
    #        status.Data_gen.size[1],2]))
        p.particles.append(par)
        p.num_particles+=1
        k=k+1
    #Initializes conectivity radius

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
    #mouse=[status.mouse_frame1[0],status.mouse_frame1[1]]
    #status.mouse_frame2=cd.coord2vector(mouse,frame1,frame2)
    #print(len(status.objects))
    #print(status.mouse_frame2)
#    positions=[[300,300],[400,400]]
    #pygame.draw.aaline(Display,white,positions[0],positions[1],True)
    for i in range(len(status.objects)-1):
        px_o=Lx_o+i*ddx
        px_f=Lx_o+(i+1)*ddx
        #print(status.objects[i].objects[0].objects[0].num_particles)
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
