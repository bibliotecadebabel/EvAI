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
from DAO import GeneratorFromCIFAR
from DNA_Graph import DNA_Graph
from DNA_Phase_space_f_ac import DNA_Phase_space
from Dynamic_DNA_f import Dynamic_DNA
from utilities.Abstract_classes.classes.torch_stream_ac import TorchStream
from utilities.Abstract_classes.classes.positive_random_selector import(
    centered_random_selector as Selector)
import children.pytorch.Network as nw
from DNA_conditions import max_layer,max_filter
from DNA_creators import Creator_from_selection as Creator
from Dyamic_DNA_f_methods import update_from_select_09  as space_updater
from Dyamic_DNA_f_methods import (
    update_velocity_mobility as velocity_updater)
import time
from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising as Alai)

import os
from DAO.database.dao import TestDAO, TestResultDAO, TestModelDAO


update_force_field=None

class Status():
    def __init__(self, display_size=None):
        self.dt_Max=0.01
        self.dt_min=0.0001
        self.max_iter=250
        self.max_layer=7
        self.max_filter=51
        self.update_force_field=update_force_field

        self.experiment_name='experiment_cifar'
        self.save_space_period=200
        self.save_net_period=1000
        self.save2database=True

        self.log_size=200
        self.min_log_size=100
        self.S=50
        self.cuda=False
        self.typos_version='clone'
        self.restart_period=200
        self.Alai=None
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
        self.S=32
        self.Comp=2
        self.clear_period=200
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
        self.Selector_creator=None
        self.Selector=None
        x=32
        y=32
        self.Center=None



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

    def print_accuracy(self):
        phase_space=self.Dynamics.phase_space
        center=phase_space.center()
        if center:
            stream=self.stream
            net=stream.get_net(center)
            net.generateEnergy(self.Data_gen)
            print(f'The acurrarcy is : {net.getAcurracy()}')
            time.sleep(4)
        pass



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
    status.Alai=Alai(min=status.dt_min,
         max=status.dt_Max,
            max_time=status.restart_period)
    status.Data_gen=GeneratorFromCIFAR.GeneratorFromCIFAR(
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
    center=status.Center
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
    status.stream=stream
    Phase_space=DNA_Phase_space(space,
        stream=stream)
    Dynamics=Dynamic_DNA(space,Phase_space,status.dx,
        Creator=creator,Selector=selector,
        update_velocity=velocity_updater,
        update_space=space_updater,version=version,
        mutation_coefficient=status.mutation_coefficient,
        clear_period=status.clear_period,
        update_force_field=status.update_force_field,
        )
    Phase_space.create_particles(status.n)
    Phase_space.beta=status.beta
    Phase_space.alpha=status.alpha
    Phase_space.influence=status.influence
    status.Dynamics=Dynamics
    status.objects=Dynamics.objects




def run(status):
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

    test_id = 0
    testDao = TestDAO.TestDAO()
    testResultDao = TestResultDAO.TestResultDAO()
    testModelDao = TestModelDAO.TestModelDAO()
    print("cuda=", status.cuda)
    if status.save2database == True:
        test_id = testDao.insert(testName=status.experiment_name, periodSave=status.save_space_period, dt=status.dt_Max,
                                          total=status.max_iter, periodCenter=1)
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
            if k % 100 == 0:
                status.print_accuracy()
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
            if k % status.restart_period == 0:
                status.Alai.time=0

            if k % 20 ==0:
                print(f'The nunber of active nents is : {len(status.stream.Graph.key2node)}')
                time.sleep(3)


            if status.save2database == True:

                if k % status.save_space_period == status.save_space_period - 1:
                    dna_graph = status.Dynamics.phase_space.DNA_graph
                    testResultDao.insert(idTest=test_id, iteration=k+1, dna_graph=dna_graph)

                if k % status.save_net_period == status.save_net_period - 1:
                    saveModel(status, k+1, testModelDao, test_id)
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

def saveModel(status, k, testModelDao, test_id):
    fileName = str(test_id)+"_"+status.experiment_name+"_model_"+str(k)
    final_path = os.path.join("saved_models","product_database",fileName)
    stream=status.Dynamics.phase_space.stream
    center=status.Dynamics.phase_space.center()
    if center:
        net=stream.get_net(center)
        net.saveModel(final_path)
        net.generateEnergy(status.Data_gen)
        testModelDao.insert(idTest=test_id, dna=str(net.adn),iteration=k, fileName=fileName, model_weight=net.getAcurracy())

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
