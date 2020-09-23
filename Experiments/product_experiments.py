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
import children.net2.Network as nc
from Geometric.Graphs.DNA_Graph import DNA_Graph
from utilities.Abstract_classes.classes.torch_stream_ac import TorchStream
from utilities.Abstract_classes.classes.positive_random_selector import(
    centered_random_selector as Selector)
from Geometric.Conditions.DNA_conditions import max_layer,max_filter
from Geometric.Creators.DNA_creators import Creator_from_selection as Creator
import time
import Geometric.Conditions.DNA_conditions as DNA_conditions
from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising as Alai)
import utilities.ExperimentSettings as ExperimentSettings
import utilities.AugmentationSettings as AugmentationSettings
import utilities.Augmentation as Augmentation_Utils
import const.versions as directions_version
import tests_scripts.test_DNAs as DNAs

def run_cifar_user_input_bidi(test_name,mutations_actions, const_direction_version):
    
    import Geometric.Products.Product_f_cifar_save_2 as program
    #import Product_f_cifar_save_nm as program

    save=True#bool(input('Insert any input to save'))
    status=program.Status()

    status.max_layer_conv2d = 11
    status.max_filter = 530
    status.max_filter_dense = 270
    status.max_kernel_dense = 9
    status.max_pool_layer = 4
    status.max_parents = 2

    list_conditions={DNA_conditions.max_filter : status.max_filter,
            DNA_conditions.max_filter_dense : status.max_filter_dense,
            DNA_conditions.max_kernel_dense : status.max_kernel_dense,
            DNA_conditions.max_layer : status.max_layer_conv2d,
            DNA_conditions.min_filter : 3,
            DNA_conditions.max_pool_layer : status.max_pool_layer,
            DNA_conditions.max_parents : status.max_parents,
            DNA_conditions.no_con_last_layer : 1,
            }

    def condition(DNA):
        return DNA_conditions.dict2condition(DNA,list_conditions)

    def dropout_function(base_p, total_layers, index_layer, isPool=False):

        value = 0
        if index_layer != 0 and isPool == False:
            value = 0.05

        if index_layer == total_layers - 2:
            value = 0.05

        return value


    settings = ExperimentSettings.ExperimentSettings()


    settings.version = const_direction_version#directions_version.POOL_VERSION
    settings.dropout_function = dropout_function
    settings.eps_batchorm = 0.001
    settings.momentum = 0.9
    custom=False#bool(input('No input for defaults :'))
    if custom == True :
        ENABLE_ACTIVATIO = int(input("Enable last layer activation? (1 = yes, 0 = no): "))
        ENABLE_LAST_ACTIVATION = int(input("Enable last layer activation? (1 = yes, 0 = no): "))
        ENABLE_AUGMENTATION = int(input("Enable Data augmentation? (1 = yes, 0 = no): "))
        ENABLE_TRACK = int(input("Enable tracking var/mean batchnorm? (1 = yes, 0 = no): "))
        settings.dropout_value = float(input("dropout value: "))
        settings.weight_decay = float(input('weight_decay: '))
    else:
        ENABLE_ACTIVATION = 1
        ENABLE_LAST_ACTIVATION = 1
        ENABLE_AUGMENTATION = 1
        ENABLE_TRACK = 1
        settings.dropout_value = 0.05
        settings.weight_decay = 0.0005


    value = True
    if ENABLE_ACTIVATION  == 0:
        value = False
    settings.enable_activation = value

    # ENABLE_LAST_ACTIVATION, enable/disable last layer relu


    value = False
    if ENABLE_LAST_ACTIVATION == 1:
        value = True
    settings.enable_last_activation = value

    # ENABLE_AUGMENTATION, enable/disable data augmentation


    value = True
    if ENABLE_AUGMENTATION == 0:
        value = False
    ENABLE_AUGMENTATION = value

    settings.enable_augmentation=value


    # ALLOW TRACK BATCHNORM

    value = True
    if ENABLE_TRACK == 0:
        value = False
    settings.enable_track_stats = value

    status.S=64
    e =  50000 / status.S
    status.iterations_per_epoch = math.ceil(e)

    status.condition=condition
    status.dt_Max=0.05
    status.dt_min=0.0000001
    status.clear_period=200000
    status.max_iter=400000
    status.restart_period=18*status.iterations_per_epoch
    status.max_layer=8
    #status.max_filter=51
    from utilities.Abstract_classes.classes.uniform_random_selector_2 import (
        centered_random_selector as Selector)
    status.mutations=mutations_actions
    status.num_actions=8#int(input("num_actions : "))

    status.Selector_creator=Selector
    status.log_size=200#int(input("Log size : "))
    status.min_log_size=100
    status.version=settings.version
    status.cuda=True#bool(input("Any input for cuda : "))

    settings.evalLoss = True#bool(input("Any input to activate EvalLoss : "))
    '''
    augSettings = AugmentationSettings.AugmentationSettings()
    dict_transformations = {
        augSettings.baseline_customRandomCrop : True,
        augSettings.randomHorizontalFlip : True,
        augSettings.randomErase_1 : True
    }

    transform_compose = augSettings.generateTransformCompose(dict_transformations, False)
    settings.transformations_compose = transform_compose
    '''

    settings.ricap = Augmentation_Utils.Ricap(beta=0.3)

    settings.cuda = status.cuda



    status.mutation_coefficient=1#float(input("mutation_coefficient : "))
    if save:
        status.experiment_name=test_name#input("insert experiment name : ")
        status.save_space_period=200#int(input("save_space_period : "))
        status.save_net_period=600#int(input("save_space_net_period : "))
    status.save2database=save
    x=32
    y=32

    status.Center = DNAs.DNA_base
    status.settings=settings
    program.run(status)

def run_cifar_user_input_bidi_back_up(save = False):
    import Product_f_cifar_save_2 as program
    status=program.Status()
    status.dt_Max=0.01
    status.dt_min=0.0001
    status.clear_period=200000
    status.max_iter=20001
    status.restart_period=200
    status.max_layer=8
    status.max_filter=51
    from utilities.Abstract_classes.classes.centered_random_selector_bidi import(
        centered_random_selector as Selector)
    status.Selector_creator=Selector
    status.log_size=int(input("Log size : "))
    status.min_log_size=100
    status.S=int(input("Batch size : "))
    status.cuda=True
    status.mutation_coefficient=float(input("mutation_coefficient : "))
    if save:
        status.experiment_name=input("insert experiment name : ")
        status.save_space_period=int(input("save_space_period : "))
        status.save_net_period=int(input("save_space_net_period : "))



    status.save2database=save
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

def run_local_ac():
    import Product_f as program
    status=program.Status()
    from Dyamic_DNA_f_methods import (
        update_force_field_ac as update_force_field)
    status.update_force_field=None
    status.dt_Max=0.1
    status.dt_min=0.01
    status.clear_period=200000
    status.max_iter=2001
    status.Alai_creator=Alai
    from utilities.Abstract_classes.classes.centered_random_selector_bidi import(
        centered_random_selector as Selector)
    status.Selector_creator=Selector
    status.log_size=50
    status.S=32
    status.min_log_size=25
    status.cuda=False
    status.restart_period=50
    status.mutation_coefficient=float(input('Mutation : '))
    status.experiment_name='experiment_test_2'
    status.save_space_period=100
    status.save_net_period=200
    status.save2database=False
    program.run(status)



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

def run_cifar_user_input_no_save():
    import Product_f_cifar_save_2 as program
    status=program.Status()
    status.dt_Max=0.1
    status.dt_min=0.00001
    status.clear_period=200
    status.max_iter=40001
    status.restart_period=200
    status.max_layer=10
    status.max_filter=51
    from utilities.Abstract_classes.classes.centered_random_selector_bidi import(
        centered_random_selector as Selector)
    status.Selector_creator=Selector
    status.log_size=int(input("Log size : "))
    status.min_log_size=100
    status.S=int(input("Batch size : "))
    status.cuda=True
    status.mutation_coefficient=float(input("mutation_coefficient : "))
    status.experiment_name="no_name"
    status.clear_period=200
    status.save_space_period=2000
    status.save_net_period=4000
    status.save2database=False
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





def run_cifar_user_input_bidi_old():
    import Product_f_cifar_save_2 as program
    status=program.Status()
    status.dt_Max=0.1
    status.dt_min=0.00001
    status.clear_period=200000
    status.max_iter=20001
    status.restart_period=200
    self.max_layer=5
    self.max_filter=51
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
    status.save2database=False
    x=32
    y=32
    self.Center=((-1,1,3,x,y),
            (0,3, 15, 3 , 3),
            (0,18, 15, 3,  3),
            (0,33, 60, x, y),
            (1, 50,10),
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


def run_local_bidirect():
    import Product_f as program
    status=program.Status()
    status.dt_Max=0.0001
    status.Alai=Alai(max_time=200,)


    status.dt_Max=0.0001
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
    status.clear_period=20
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
