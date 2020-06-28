import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Safe as sf
import utilities.P_trees as tr
import numpy as np
import Geometric.TangentPlane as tplane
import utilities.Graphs as gr
import math
import Transfer.Transfer as tran
import children.Data_generator as dgen
import children.Interfaces as Inter
import children.Operations as Op
from DAO import GeneratorFromCIFAR
from Geometric.Graphs.DNA_Graph import DNA_Graph
from Geometric.Space.DNA_Phase_space_f_disk import DNA_Phase_space
from Geometric.Dynamic.Dynamic_DNA_f import Dynamic_DNA
from utilities.Abstract_classes.classes.torch_stream_disk import TorchStream
from utilities.Abstract_classes.classes.positive_random_selector import(
    centered_random_selector as Selector)
import children.pytorch.NetworkDendrites as nw
from Geometric.Conditions.DNA_conditions import max_layer,max_filter
from Geometric.Creators.DNA_creators import Creator_from_selection as Creator
from Geometric.Dynamic.Dyamic_DNA_f_methods import update_from_select_09  as space_updater
from Geometric.Dynamic.Dyamic_DNA_f_methods import (
    update_velocity_mobility as velocity_updater)
import time
from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising as Alai)

import os
from DAO.database.dao import TestDAO, TestResultDAO, TestModelDAO
import const.training_type as TrainingType
import const.file_names as FileNames
import utilities.NetworkStorage as NetworkStorage
from Geometric.Dynamic.Dyamic_DNA_f_methods import ( update_force_field_ac
    as update_force_field )
import utilities.FileManager as FileManager
import utilities.CheckpointModel as CheckPoint
import json

class Status():
    def __init__(self, display_size=None):
        self.dt_Max=0.05
        self.dt_min=0.0001
        self.max_iter=250
        self.max_layer=7
        self.update_force_field=update_force_field
        self.condition=None
        self.experiment_name='experiment_cifar'
        self.save_space_period=200
        self.save_net_period=1000
        self.save2database=True
        self.threads=int(input('Threads : '))
        self.mutations=None
        self.num_actions=4
        self.settings=None

        self.log_size=200
        self.min_log_size=100
        self.S=50
        self.cuda=False
        self.typos_version='clone'
        self.version=None
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
        self.iterations_per_epoch = 0

        ## CONDITIONS
        self.max_layer_conv2d = 0
        self.max_filter_dense = 0
        self.max_filter=51
        self.max_kernel_dense = 0
        self.max_pool_layer = 0
        self.max_parents = 2



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
    settings=status.settings
    status.Alai=Alai(min=status.dt_min,
         max=status.dt_Max,
            max_time=status.restart_period)
    status.Data_gen=GeneratorFromCIFAR.GeneratorFromCIFAR(
    status.Comp, status.S, cuda=status.cuda, threads=status.threads,
        dataAugmentation=settings.enable_augmentation, transforms_mode=settings.transformations_compose)
    status.Data_gen.dataConv2d()
    dataGen=status.Data_gen
    x = dataGen.size[1]
    y = dataGen.size[2]
    condition=status.condition
    mutations=status.mutations
    version=status.version
    center=status.Center
    num_actions=status.num_actions
    selector=status.Selector_creator(condition=condition,
        directions=version, num_actions=num_actions,
        mutations=mutations)
    status.Selector=selector
    creator=status.Creator
    selector.update(center)
    actions=selector.get_predicted_actions()
    space=DNA_Graph(center,1,(x,y),condition,actions,
        version,creator)
    if status.Alai:
        stream=TorchStream(status.Data_gen,status.log_size,
            min_size=status.min_log_size,
            Alai=status.Alai,status=status)
    else:
        stream=TorchStream(status.Data_gen,status.log_size,
            min_size=status.min_log_size,status=status)
    status.stream=stream
    Phase_space=DNA_Phase_space(space,
        stream=stream,status=status)
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

def countLayers(center):
    count = 0

    for layer in center:

        if layer[0] == 0:
            count += 1

    return count

def run(status):

    fileManager = FileManager.FileManager()
    fileManager.setFileName(FileNames.SAVED_MODELS)
    fileManager.writeFile("")

    status.Transfer=tran.TransferRemote(status,
        'remote2local.txt','local2remote.txt')
    print(f'status.settings is {status.settings}')

    create_objects(status)
    print('The value of typos after loading is')
    print(status.typos)
    print("objects created")
    status.print_DNA()
    status.Transfer.un_load()
    status.Transfer.write()
    k=0

    print("max convolution layers: ", status.max_layer_conv2d)

    settings = status.settings

    network = nw.Network(status.Center,cudaFlag=settings.cuda,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
            enable_activation=settings.enable_activation,
            enable_track_stats=settings.enable_track_stats,
            dropout_value=settings.dropout_value,
            dropout_function=settings.dropout_function,
            enable_last_activation=settings.enable_last_activation,
            version=settings.version, eps_batchnorm=settings.eps_batchorm
            )

    print("starting pre-training")
    dt_array=status.Alai.get_increments(20*status.iterations_per_epoch)

    network.iterTraining(dataGenerator=status.Data_gen,
                    dt_array=dt_array, ricap=settings.ricap, evalLoss=settings.evalLoss)

    status.stream.add_node(network.adn)
    status.stream.link_node(network.adn, network)

    last_center_dna = None
    while k<status.max_iter:
 
        status.active=True

        if status.active:
            update(status)

            print(f'The iteration number is: {k}')
            status.print_predicted_actions()

            if status.Alai:
                status.Alai.update()

            center_dna = status.Dynamics.phase_space.center()
            layers_count = 0

            if center_dna is not None:
                layers_count = countLayers(center_dna)

                if center_dna != last_center_dna:
                    last_center_dna = center_dna
                    save_model(dna=center_dna, alaiTime=status.Alai.computeTime(), fileManager=fileManager)

            print("current layers: ", layers_count)
            
            if layers_count >= status.max_layer_conv2d:
                print("STOPPED MAX LAYERS: ", layers_count)
                break
        k=k+1

def save_model(dna, alaiTime, fileManager):

    print("Saving DNA Model")
    checkpoint = CheckPoint.CheckPointModel(alaiTime, dna)
    value = str(json.dumps(checkpoint.__dict__))
    fileManager.appendFile(value)
    



