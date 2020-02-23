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
from DNA_Graph import DNA_Graph
from DNA_Phase_space import DNA_Phase_space
from Dynamic_DNA import Dynamic_DNA
from utilities.Abstract_classes.classes.torch_stream import TorchStream
import children.pytorch.Network as nw
import time

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
        self.Dynamics=None

def potential(x,status=None):
    return node_energy(status.objects[x],status)

def r_potential(x):
    return -1/x

def interaction(r,status):
    return (200*r/(2**status.dx))**(status.alpha)/abs(status.alpha-1)

def update(status):
    status.Dynamics.update()

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

def create_objects(status):
    status.Data_gen=GeneratorFromImage.GeneratorFromImage(
    status.Comp, status.S, cuda=False)
    status.Data_gen.dataConv2d()
    dataGen=status.Data_gen
    x = dataGen.size[1]
    y = dataGen.size[2]
    ks=[2]
    center=((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
    space=DNA_Graph(center,status.dx,(x,y))
    Phase_space=DNA_Phase_space(space)
    Dynamics=Dynamic_DNA(space,Phase_space)
    Phase_space.create_particles(100)
    Phase_space.beta=status.beta
    status.Dynamics=Dynamics
    status.objects=Dynamics.objects

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
print("objects created")
status.Transfer=tran.TransferRemote(status,
    'remote2local.txt','local2remote.txt')
status.Transfer.un_load()
status.Transfer.write()
k=0
#update(status)
while False:
    update(status)
    status.Transfer.un_load()
    status.Transfer.write()
    transfer=status.Transfer.status_transfer
    k=k+1
    pass
while False:
    status.Transfer.readLoad()
    if status.active:
        update(status)
#        print_nets(status)
#        time.sleep(0.5)
    else:
        print('inactive')
    k=k+1


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
