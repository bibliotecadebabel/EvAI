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
from DNA_Phase_space_f_ac import DNA_Phase_space
from Dynamic_DNA_f import Dynamic_DNA
from utilities.Abstract_classes.classes.torch_stream_ac import TorchStream
from utilities.Abstract_classes.classes.positive_random_selector import(
    centered_random_selector as Selector)

import children.pytorch.Network as nw
from DNA_conditions import max_layer,max_filter
from DNA_creators import Creator_from_selection as Creator
from Dyamic_DNA_f_methods import update_from_select_09 as space_updater
from Dyamic_DNA_f_methods import (
    update_velocity_mobility as velocity_updater)

import time
from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising as Alai)

class Status():
    def __init__(self, display_size=None):
        self.dt_Max=0.001
        self.dt_min=0.0001
        self.max_iter=10000
        self.max_layer=5
        self.max_filter=60
        self.log_size=50
        self.min_log_size=30
        self.cuda=bool(input("Insert flag for cuda"))
        self.typos_version='clone'
        #self.Alai=None
        self.Alai=Alai(min=self.dt_min
            , max=self.dt_Max,
                max_time=self.max_iter+self.log_size)
        self.max_layer=5
        self.max_filter=60
        self.log_size=50
        self.min_log_size=30
        self.typos=((1,0,0,0),(0,0,1,1),(0,1,0,0))
        self.dt = 0.1
        self.tau=0.01
        self.n = 1000
        self.r=3
        self.dx = 1
        self.L = 1
        self.beta = 2
        self.alpha = 50
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
        self.S=200
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
        self.Creator=Creator
        self.Selector_creator=Selector
        self.Selector=None



        #self.typos=(0,(0,0,1,1))
        self.influence=2
    def print_DNA(self):
        phase_space=self.Dynamics.phase_space
        DNA_graph=phase_space.DNA_graph
        DNA_graph.imprimir()
    def print_energy(self):
        phase_space=self.Dynamics.phase_space
        stream=phase_space.stream
        stream.imprimir()
    def print_signal(self):
        phase_space=self.Dynamics.phase_space
        stream=phase_space.stream
        stream.print_signal()
    def print_particles(self):
        Dynamics=self.Dynamics
        Dynamics.print_particles()
    def print_difussion_filed(self):
        Dynamics=self.Dynamics
        phase_space=Dynamics.phase_space
        phase_space.print_diffussion_field()
    def print_max_particles(self):
        Dynamics=self.Dynamics
        phase_space=Dynamics.phase_space
        phase_space.print_max_particles()
    def print_predicted_actions(self):
        selector=self.Selector
        actions=selector.get_predicted_actions()
        print(f'The predicted actions are: {actions}')



def potential(x,status=None):
    return node_energy(status.objects[x],status)

def r_potential(x):
    return -1/x

def interaction(r,status):
    return (200*r/(2**status.dx))**(status.alpha)/abs(status.alpha-1)

def update(status):
    status.Dynamics.update()
    status.Transfer.status=status
    status.Transfer.update()

def initialize_parameters(self):
    display_size=[1000,500]
    self.dt=0.01
    self.n=1000
    self.dx=1
    self.L=1
    self.beta=2
    self.alpha=50
    status.influence=2.5
    self.center=.5
    self.std_deviation=1
    self.Potantial=potential
    self.Interaction=interaction
    self.display_size=display_size
    self.dx = 2


def create_objects(status):
    status.Data_gen=GeneratorFromImage.GeneratorFromImage(
    status.Comp, status.S, cuda=status.cuda)
    status.Data_gen.dataConv2d()
    dataGen=status.Data_gen
    x = dataGen.size[1]
    y = dataGen.size[2]
    max_layers=status.max_layer
    max_filters=status.max_filter
    def condition(DNA):
        return max_filter(max_layer(DNA,max_layers),max_filters)
    version=status.typos_version
    """
    center=((-1,1,3,x,y),
            (0,3, 4, x, y),
            (1, 4, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2))"""
    center=((-1,1,3,x,y),
            (0,3, 15, 3 , 3),
            (0,18, 15, 3,  3),
            (0,33, 15, x, y),
            (1, 15,2),
             (2,),
            (3,-1,0),
            (3,0,1),(3,-1,1),
            (3,1,2),(3,0,2),(3,-1,2),
            (3,2,3),
            (3,3,4))
    selector=status.Selector_creator(condition=condition,
        directions=version)
    status.Selector=selector
    creator=status.Creator
    selector.update(center)
    actions=selector.get_predicted_actions()
    space=DNA_Graph(center,1,(x,y),condition,actions,
        version,creator)
    if status.Alai:
        stream=TorchStream(status.Data_gen,status.log_size,
            min_size=status.min_log_size,
            Alai=status.Alai)
    else:
        stream=TorchStream(status.Data_gen,status.log_size,
            min_size=status.min_log_size)
    Phase_space=DNA_Phase_space(space,
        stream=stream)
    Dynamics=Dynamic_DNA(space,Phase_space,status.dx,
        Creator=creator,Selector=selector,
        update_velocity=velocity_updater,
        update_space=space_updater,version=version)
    Phase_space.create_particles(status.n)
    Phase_space.beta=status.beta
    Phase_space.alpha=status.alpha
    Phase_space.influence=status.influence
    status.Dynamics=Dynamics
    status.objects=Dynamics.objects

status=Status()
initialize_parameters(status)
status.Transfer=tran.TransferRemote(status,
    'remote2local.txt','local2remote.txt')
#status.Transfer.readLoad()
create_objects(status)
print('The value of typos after loading is')
print(status.typos)
print("objects created")
status.print_DNA()
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
while k<status.max_iter:
    #\begin{with gui}
    #status.Transfer.readLoad()
    #\end{with gui}
    #\begin{wituhout gui}
    status.active=True
    #\end{without gui}
    if status.active:
        update(status)
        print(f'The iteration number is: {k}')
        #status.print_energy()
        status.print_predicted_actions()
        if status.Alai:
            status.Alai.update()
        #status.print_particles()
        #status.print_particles()
        #status.print_max_particles()
        #print(status.typos)
        #status.print_signal()
        #status.print_difussion_filed()
#        print_nets(status)
#        time.sleep(0.5)
    else:
        #print('inactive')
        pass
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
