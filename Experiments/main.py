import utilities.ExperimentSettings as ExperimentSettings
import utilities.AugmentationSettings as AugmentationSettings
import utilities.Augmentation as Augmentation_Utils
import const.versions as directions_version
import tests_scripts.test_DNAs as DNAs
import time
import Geometric.Conditions.DNA_conditions as DNA_conditions
from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising as Alai)
import math
import const.versions as directions_version

import Geometric.Products.Product_main as program


def run_cifar_user_input_bidi():

    status=program.Status()

    status.max_layer_conv2d = int(input("Enter max convolution layers model: "))
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


    settings.version = directions_version.CONVEX_VERSION
    settings.dropout_function = dropout_function
    settings.eps_batchorm = 0.001
    settings.momentum = 0.9
    settings.enable_activation = True
    settings.enable_last_activation = True
    settings.enable_augmentation= True
    settings.enable_track_stats = True
    settings.dropout_value = 0.05
    settings.weight_decay = 0.0005
    settings.evalLoss = True

    # Batch size
    status.S = 64 
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
    status.mutations=(
    (1,0,0,0),(1,0,0,0),
    (0,1,0,0),(0,1,0,0),
    (4,0,0,0),
    (0,0,1),(0,0,-1),
    (0,0,1,1),(0,0,-1,-1),
    (0,0,2)
    )
    status.num_actions=int(input("num_actions : "))

    status.Selector_creator= Selector
    status.log_size = 200
    status.min_log_size = 100
    status.version= directions_version.CONVEX_VERSION
    status.cuda = True
    
    augSettings = AugmentationSettings.AugmentationSettings()

    dict_transformations = {
        augSettings.baseline_customRandomCrop : True,
        augSettings.randomHorizontalFlip : True,
        augSettings.randomErase_1 : True
    }

    transform_compose = augSettings.generateTransformCompose(dict_transformations, False)
    settings.transformations_compose = transform_compose
    settings.ricap = Augmentation_Utils.Ricap(beta=0.3)

    settings.cuda = status.cuda
    status.mutation_coefficient=float(input("mutation_coefficient : "))        
    status.save2database= False

    # DNA BASE
    status.Center = DNAs.DNA_base
    
    status.settings=settings
    program.run(status)
