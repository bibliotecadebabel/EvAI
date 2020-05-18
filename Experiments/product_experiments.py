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

#from Product_f_cifar import Status as program_cf

def run_cifar_user_input_save():
    import Product_f_cifar_save_2 as program
    status=program.Status()
    status.dt_Max=0.01
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=200000
    status.log_size=int(input("Log size : "))
    status.min_log_size=100
    status.S=int(input("Batch size : "))
    status.cuda=True
    status.mutation_coefficient=float(input("mutation_coefficient : "))
    status.experiment_name=input("experiment_name : ")
    status.save_space_period=int(input("save_space_period : "))
    status.save_net_period=int(input("save_space_net_period : "))
    status.save2database=True
    program.run(status)


def run_cifar_user_input():
    import Product_f_cifar_save_2 as program
    status=program.Status()
    status.dt_Max=0.01
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=20000
    status.log_size=int(input("Log size : "))
    status.min_log_size=100
    status.S=int(input("Batch size : "))
    status.cuda=True
    status.mutation_coefficient=float(input("mutation_coefficient : "))
    status.experiment_name=float(input("experiment_name : "))
    status.save_space_period=float(input("save_space_period : "))
    status.save_net_period=float(input("save_space_net_period : "))
    status.save2database=float(input("mutation_coefficient : "))
    program.run(status)


def run_cifar_small_batch():
    import Product_f_cifar_save_2 as program
    status=program.Status()
    status.dt_Max=0.01
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=20000
    status.log_size=200
    status.min_log_size=100
    status.S=32
    status.cuda=True
    status.mutation_coefficient=10
    status.experiment_name='experiment_cifar_10mutation'
    status.save_space_period=100
    status.save_net_period=200
    status.save2database=False
    program.run(status)

def run_slow_ncf():
    import Product_f as program
    status=program.Status()
    status.dt_Max=0.0001
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=201
    status.log_size=200
    status.S=32
    status.min_log_size=50
    status.cuda=False
    status.mutation_coefficient=1
    status.experiment_name='experiment_test_2'
    status.save_space_period=100
    status.save_net_period=200
    status.save2database=False
    program.run(status)

def run_test():
    status=program.Status()
    program.run(status)


def import_test():
    program=program_0()
    print('done')
