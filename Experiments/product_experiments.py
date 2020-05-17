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
import Product_f as program
#from Product_f_cifar import Status as program_cf


def run_slow_ncf():
    status=program.Status()
    status.dt_Max=0.0001
    status.dt_min=0.00001
    status.clear_period=200
    self.log_size=50
    self.min_log_size=3
    status.cuda=False
    status.mutation_coefficient=0.1
    status.experiment_name='experiment'
    status.save_space_period=2000
    status.save_net_period=10000
    status.save2database=False
    program.run(status)

def run_test():
    status=program.Status()
    program.run(status)


def import_test():
    program=program_0()
    print('done')
