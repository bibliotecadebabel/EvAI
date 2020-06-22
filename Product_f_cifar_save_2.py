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
from DAO import GeneratorFromCIFAR
from DNA_Graph import DNA_Graph
from DNA_Phase_space_f_disk import DNA_Phase_space
from Dynamic_DNA_f import Dynamic_DNA
from utilities.Abstract_classes.classes.torch_stream_disk import TorchStream
from utilities.Abstract_classes.classes.positive_random_selector import(
    centered_random_selector as Selector)
import children.pytorch.NetworkDendrites as nw
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
import const.training_type as TrainingType
import utilities.NetworkStorage as NetworkStorage

update_force_field=None

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
    status.Transfer=tran.TransferRemote(status,
        'remote2local.txt','local2remote.txt')
    #status.Transfer.readLoad()
    print(f'status.settings is {status.settings}')
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

    print("max layers: ", status.max_layer_conv2d)
    loaded_network = bool(input("any input to run loaded network"))
    print("Loaded network: ", loaded_network)
    
    settings = status.settings

    if loaded_network == False:
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

        print("iterations per epoch = ", status.iterations_per_epoch)
        dt_array=status.Alai.get_increments(20*status.iterations_per_epoch)

        if status.save2database == True:
            test_id = testDao.insert(testName=status.experiment_name, dt=status.dt_Max, dt_min=status.dt_min, batch_size=status.S,
                    max_layers=status.max_layer_conv2d, max_filters=status.max_filter, max_filter_dense=status.max_filter_dense,
                    max_kernel_dense=status.max_kernel_dense, max_pool_layer=status.max_pool_layer, max_parents=status.max_parents)

        network.iterTraining(dataGenerator=status.Data_gen,
                        dt_array=dt_array, ricap=settings.ricap, evalLoss=settings.evalLoss)

    else:
        
        path = os.path.join("saved_models","product_database", "5_test_final_experiment_dnabase2_model_7107")
        network = NetworkStorage.loadNetwork(fileName=None, settings=settings, path=path)
        network.generateEnergy(status.Data_gen)
        acc = network.getAcurracy()
        print("Acc loaded network: ", acc)
        time.sleep(2)

        if status.save2database == True:
            test_id = testDao.insert(testName=status.experiment_name, dt=status.dt_Max, dt_min=status.dt_min, batch_size=status.S,
                    max_layers=status.max_layer_conv2d, max_filters=status.max_filter, max_filter_dense=status.max_filter_dense,
                    max_kernel_dense=status.max_kernel_dense, max_pool_layer=status.max_pool_layer, max_parents=status.max_parents)


    status.stream.add_node(network.adn)
    status.stream.link_node(network.adn, network)

    if status.save2database == True and loaded_network == False:
        dna_graph = status.Dynamics.phase_space.DNA_graph
        testResultDao.insert(idTest=test_id, iteration=0, dna_graph=dna_graph, current_alai_time=status.Alai.computeTime(), 
                                reset_count=status.Alai.reset_count)

        saveModel(status, 0, testModelDao, test_id, TrainingType.PRE_TRAINING)  
                            
    #update(status)
    while False:
        update(status)
        status.Transfer.un_load()
        status.Transfer.write()
        transfer=status.Transfer.status_transfer
        k=k+1
        pass

    L_1 = 1
    L_2 = 1

    save_6_layers = True
    save_17_layers = True
    save_18_layers = True
    save_19_layers = True
    save_24_layers = True
    save_25_layers = True
    save_26_layers = True
    save_27_layers = True
    save_30_layers = True
    save_33_layers = True
    save_36_layers = True
    save_39_layers = True
    save_42_layers = True
    save_45_layers = True
    save_48_layers = True
    save_51_layers = True

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
            #if k % 20 == 0:
            #    status.print_accuracy()
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
    #        time.sleep(0.5)0

            center_dna = status.Dynamics.phase_space.center()
            layers_count = 0
            if center_dna is not None:
                layers_count = countLayers(center_dna)

            print("current layers: ", layers_count)
            if status.save2database == True:

                if status.Alai.computeTime() >= L_1*status.save_space_period:
                    print("saving space: ", L_1)
                    L_1 += 1
                    dna_graph = status.Dynamics.phase_space.DNA_graph
                    testResultDao.insert(idTest=test_id, iteration=k+1, dna_graph=dna_graph, current_alai_time=status.Alai.computeTime(), 
                                            reset_count=status.Alai.reset_count)

                if status.Alai.computeTime() >= L_2*status.save_net_period:
                    print("saving model: ", L_2)
                    L_2 += 1
                    saveModel(status, k+1, testModelDao, test_id, TrainingType.MUTATION)

                if layers_count >= 6 and save_6_layers == True:
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
                    save_6_layers = False

                elif layers_count >= 17 and save_17_layers == True:
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
                    save_17_layers = False

                elif layers_count >= 18 and save_18_layers == True:
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
                    save_18_layers = False

                elif layers_count >= 19 and save_19_layers == True:
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
                    save_19_layers = False 

                elif layers_count >= 24 and save_24_layers == True:
                    save_24_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)

                elif layers_count >= 25 and save_25_layers == True:
                    save_25_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
                
                elif layers_count >= 26 and save_26_layers == True:
                    save_26_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)

                elif layers_count >= 27 and save_27_layers == True:
                    save_27_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
    
                elif layers_count >= 30 and save_30_layers == True:
                    save_30_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
    
                elif layers_count >= 33 and save_33_layers == True:
                    save_33_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
                
                elif layers_count >= 36 and save_36_layers == True:
                    save_36_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
    
                elif layers_count >= 39 and save_39_layers == True:
                    save_39_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
    
                elif layers_count >= 42 and save_42_layers == True:
                    save_42_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
    
                elif layers_count >= 45 and save_45_layers == True:
                    save_45_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)

                elif layers_count >= 48 and save_48_layers == True:
                    save_48_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)
    
                elif layers_count >= 51 and save_51_layers == True:
                    save_51_layers = False
                    save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k)

                        
        else:
            #print('inactive')
            pass
        k=k+1

def save_checkpoint(status, testResultDao, layers_count, test_id, testModelDao, k):
    print("saving model and space, current layers: ", layers_count)  
    dna_graph = status.Dynamics.phase_space.DNA_graph
    testResultDao.insert(idTest=test_id, iteration=k+1, dna_graph=dna_graph, current_alai_time=status.Alai.computeTime(), 
                            reset_count=status.Alai.reset_count)
    saveModel(status, k+1, testModelDao, test_id, TrainingType.MUTATION) 

def saveModel(status, k, testModelDao, test_id, trainingType):
    fileName = str(test_id)+"_"+status.experiment_name+"_model_"+str(k)
    final_path = os.path.join("saved_models","product_database",fileName)
    stream=status.Dynamics.phase_space.stream
    center=status.Dynamics.phase_space.center()
    if center:
        net=stream.get_net(center)
        net.saveModel(final_path)
        #net.generateEnergy(status.Data_gen)
        #acc = net.getAcurracy()
        acc = 0
        testModelDao.insert(idTest=test_id, dna=str(net.adn),iteration=k, fileName=fileName, model_weight=acc,
                                current_alai_time=status.Alai.computeTime(), reset_count=status.Alai.reset_count, 
                                training_type=trainingType)

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
