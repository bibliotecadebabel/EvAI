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


def run_cifar_user_no_image():
    import Product_f_cifar_save_2 as program
    status=program.Status()
    x=32
    y=32
    status.Center=((-1,1,3,x,y),
            (0,3, 5, 3 , 3),
            (0,5, 5, 3,  3),
            (0,5, 120, x-4, y-4),
            (1, 120,10),
            (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4))
    status.dt_Max=0.1
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_layer=6
    status.max_filters=41
    status.max_iter=2001
    status.restart_period=200
    from utilities.Abstract_classes.classes.centered_random_selector_bidi import(
        centered_random_selector as Selector)
    status.Selector_creator=Selector
    status.log_size=int(input("Log size : "))
    status.min_log_size=100
    status.S=int(input("Batch size : "))
    status.cuda=True
    status.mutation_coefficient=float(input("mutation_coefficient : "))
    status.experiment_name='do not sve'
    status.save_space_period=100
    status.save_net_period=50
    status.save2database=False


    program.run(status)


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
    status.experiment_name=int("experiment_name : ")
    status.save_space_period=int(input("save_space_period : "))
    status.save_net_period=float(input("save_space_net_period : "))
    status.save2database=True
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

def run_cifar_user_input_bidi_save():
    import Product_f_cifar_save_2 as program
    status=program.Status()
    status.dt_Max=0.1
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=20001
    status.restart_period=200
    status.max_layer=7
    status.max_filter=41
    from utilities.Abstract_classes.classes.centered_random_selector_bidi import(
        centered_random_selector as Selector)
    status.Selector_creator=Selector
    status.log_size=int(input("Log size : "))
    status.min_log_size=100
    status.S=int(input("Batch size : "))
    status.cuda=True
    status.mutation_coefficient=float(input("mutation_coefficient : "))
    status.experiment_name="max_filtera=41_bidi"
    status.save_space_period=int(input("save_space_period : "))
    status.save_space_period=2000
    status.save_net_period=4000
    status.save2database=True
    x=32
    y=32
    status.Center=((-1,1,3,x,y),
            (0,3, 15, 3 , 3),
            (0,18, 15, 3,  3),
            (0,33, 50, x, y),
            (1, 50,10),
             (2,),
            (3,-1,0),
            (3,0,1),(3,-1,1),
            (3,1,2),(3,0,2),(3,-1,2),
            (3,2,3),
            (3,3,4))
    program.run(status)


def run_cifar_user_input_bidi():
    import Product_f_cifar_save_2 as program
    status=program.Status()
    status.dt_Max=0.1
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=20001
    status.restart_period=200
    self.max_layer=5
    self.max_filter=41
    from utilities.Abstract_classes.classes.centered_random_selector_bidi import(
        centered_random_selector as Selector)
    status.Selector_creator=Selector
    status.log_size=int(input("Log size : "))
    status.min_log_size=100
    status.S=int(input("Batch size : "))
    status.cuda=True
    status.mutation_coefficient=float(input("mutation_coefficient : "))
    status.experiment_name=intput('experiment name : ')
    status.save_space_period=2000
    status.save_net_period=4000
    status.save2database=True
    x=32
    y=32
    self.Center=((-1,1,3,x,y),
            (0,3, 15, 3 , 3),
            (0,18, 15, 3,  3),
            (0,33, 60, x, y),
            (1, 60,10),
             (2,),
            (3,-1,0),
            (3,0,1),(3,-1,1),
            (3,1,2),(3,0,2),(3,-1,2),
            (3,2,3),
            (3,3,4))
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
    status.experiment_name='do not sve'
    status.save_space_period=100
    status.save_net_period=50
    status.save2database=False
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

def run_local_no_image():
    import Product_f as program
    status=program.Status()
    from Dyamic_DNA_f_methods import (
        update_force_field_ac as update_force_field)
    status.update_force_field=update_force_field
    status.dt_Max=0.001
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=2001
    from utilities.Abstract_classes.classes.centered_random_selector_bidi import(
        centered_random_selector as Selector)
    status.Selector_creator=Selector
    status.log_size=50
    status.S=32
    status.min_log_size=25
    status.cuda=False
    status.restart_period=50
    status.mutation_coefficient=10
    status.experiment_name='experiment_test_2'
    status.save_space_period=100
    status.save_net_period=200
    status.save2database=False
    x=11
    y=11
    status.Center=((-1,1,3,x,y),
            (0,3, 5, 3 , 3),
            (0,5, 5, 3,  3),
            (0,5, 40, x-4, y-4),
            (1, 40,2),
             (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4))
    program.run(status)

def run_local_ac():
    import Product_f as program
    status=program.Status()
    from Dyamic_DNA_f_methods import (
        update_force_field_ac as update_force_field)
    status.update_force_field=update_force_field
    status.dt_Max=0.001
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=2001
    from utilities.Abstract_classes.classes.centered_random_selector_bidi import(
        centered_random_selector as Selector)
    status.Selector_creator=Selector
    status.log_size=50
    status.S=32
    status.min_log_size=25
    status.cuda=False
    status.restart_period=50
    status.mutation_coefficient=10
    status.experiment_name='experiment_test_2'
    status.save_space_period=100
    status.save_net_period=200
    status.save2database=False
    program.run(status)

def run_local_bidirect():
    import Product_f as program
    status=program.Status()
    status.dt_Max=0.1
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=2001
    from utilities.Abstract_classes.classes.centered_random_selector_bidi import(
        centered_random_selector as Selector)
    status.Selector_creator=Selector
    status.log_size=50
    status.S=32
    status.min_log_size=25
    status.cuda=False
    status.restart_period=50
    status.mutation_coefficient=1
    status.experiment_name='experiment_test_2'
    status.save_space_period=100
    status.save_net_period=200
    status.save2database=False
    program.run(status)


def run_slow_ncf():
    import Product_f as program
    status=program.Status()
    status.dt_Max=0.1
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=2001
    status.log_size=50
    status.S=32
    status.min_log_size=25
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
