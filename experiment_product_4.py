import children.pytorch.network_dendrites as nw
from Logic.commands import CommandCreateDataGen
from utilities.FileManager import FileManager as FileManager
import numpy as np
import utilities.Augmentation as Augmentation
import utilities.AugmentationSettings as AugmentationSettings
import utilities.ExperimentSettings as ExperimentSettings
import const.versions as directions_version
import utilities.MemoryManager as MemoryManager
import math

fileManager = FileManager()
fileManager.setFileName("product_4_results.txt")
fileManager.writeFile("")
fileManager.appendFile("#### RESULTADOS PRIMER Y SEGUNDO ORDEN ####")

# 5L, 6L, 7L, 8L, 9L

#### 1ST ORDER ####

# ID = 39 / FOUND AT = 200
DNA_product_4_1st_5L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 128, 256, 3, 3, 2), (0, 256, 512, 3, 3), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6))
NAME_DNA_product_4_1st_5L = "1st_order_5L"

# ID = 40 / FOUND AT = 600
DNA_product_4_1st_6L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 128, 256, 3, 3, 2), (0, 256, 256, 3, 3), (0, 256, 512, 3, 3), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 6, 7))
NAME_DNA_product_4_1st_6L = "1st_order_6L"

# ID = 1061 (test_result) / FOUND AT = 800 
DNA_product_4_1st_7L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 128, 128, 3, 3, 2), (0, 256, 256, 3, 3, 2), (0, 256, 256, 3, 3), (0, 256, 512, 3, 3), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 1, 3), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 6, 7), (3, 7, 8))
NAME_DNA_product_4_1st_7L = "1st_order_7L"

# ID = 42 / FOUND AT = 1600
DNA_product_4_1st_8L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 128, 128, 4, 4, 2), (0, 256, 256, 3, 3, 2), (0, 256, 256, 2, 2), (0, 256, 256, 4, 4, 2), (0, 512, 512, 3, 3), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 1, 3), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 4, 6), (3, 5, 6), (3, 6, 7), (3, 7, 8), (3, 8, 9))
NAME_DNA_product_4_1st_8L = "1st_order_8L"

# ID = 45 / FOUND AT = 4000
DNA_product_4_1st_9L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 128, 128, 4, 4, 2), (0, 256, 256, 3, 3, 2), (0, 256, 512, 2, 2), (0, 640, 512, 2, 2), (0, 640, 512, 3, 3, 2), (0, 1024, 512, 4, 4), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 1, 3), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 1, 5), (3, 5, 6), (3, 1, 6), (3, 4, 7), (3, 6, 7), (3, 7, 8), (3, 8, 9), (3, 9, 10))
NAME_DNA_product_4_1st_9L = "1st_order_9L"



#### 2ND ORDER ####

# ID = 161 / FOUND AT = 2346
DNA_product_4_2nd_5L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 192, 128, 3, 3, 2), (0, 128, 256, 3, 3, 2), (0, 256, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 0, 2), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6))
NAME_DNA_product_4_2nd_5L = "2nd_order_5L"

# ID = 162 / FOUND AT = 3128
DNA_product_4_2nd_6L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 192, 128, 3, 3, 2), (0, 128, 256, 3, 3, 2), (0, 256, 256, 3, 3), (0, 256, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 0, 2), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 6, 7))
NAME_DNA_product_4_2nd_6L = "2nd_order_6L"

# ID = 164 / FOUND AT = 4692
DNA_product_4_2nd_7L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 192, 128, 3, 3, 2), (0, 128, 128, 3, 3, 2), (0, 256, 256, 3, 3, 2), (0, 256, 256, 2, 2), (0, 256, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 0, 2), (3, 1, 2), (3, 2, 3), (3, 2, 4), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 6, 7), (3, 7, 8))
NAME_DNA_product_4_2nd_7L = "2nd_order_7L"

# ID = 178  / FOUND AT = 15640 
DNA_product_4_2nd_8L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 256, 4, 4, 2), (0, 320, 128, 3, 3, 2), (0, 128, 512, 2, 2, 2), (0, 256, 256, 5, 5), (0, 768, 512, 5, 5, 2), (0, 1024, 256, 2, 2), (0, 256, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 0, 2), (3, 1, 2), (3, 2, 3), (3, 2, 4), (3, 1, 4), (3, 4, 5), (3, 3, 5), (3, 5, 6), (3, 3, 6), (3, 6, 7), (3, 7, 8), (3, 8, 9))
NAME_DNA_product_4_2nd_8L = "2nd_order_8L"

# ID = 179 / FOUND AT = 16599
DNA_product_4_2nd_9L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 256, 4, 4, 2), (0, 320, 128, 3, 3, 2), (0, 128, 512, 2, 2, 2), (0, 256, 256, 5, 5), (0, 768, 512, 4, 4, 2), (0, 512, 512, 3, 3), (0, 1024, 256, 2, 2), (0, 256, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 0, 2), (3, 1, 2), (3, 2, 3), (3, 2, 4), (3, 1, 4), (3, 4, 5), (3, 3, 5), (3, 3, 6), (3, 5, 7), (3, 6, 7), (3, 7, 8), (3, 8, 9), (3, 9, 10))
NAME_DNA_product_4_2nd_9L = "2nd_order_9L"



#### BEST DNAS ####

# ID = 58  / FOUND AT = 12200
BEST_1ST_ORDER = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 64, 3, 3), (0, 64, 128, 3, 3, 2), (0, 128, 64, 3, 3), (0, 64, 64, 3, 3), (0, 128, 128, 3, 3, 2), (0, 128, 128, 3, 3), (0, 128, 256, 2, 2), (0, 256, 256, 3, 3), (0, 256, 512, 3, 3), (0, 512, 512, 3, 3, 2), (0, 512, 512, 3, 3), (0, 512, 512, 5, 5), (0, 256, 256, 3, 3), (0, 768, 128, 3, 3), (0, 256, 256, 3, 3, 2), (0, 256, 512, 4, 4), (0, 512, 512, 3, 3), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 0, 4), (3, 4, 5), (3, 3, 5), (3, 5, 6), (3, 6, 7), (3, 7, 8), (3, 8, 9), (3, 9, 10), (3, 2, 10), (3, 10, 11), (3, 11, 12), (3, 7, 13), (3, 13, 14), (3, 12, 14), (3, 14, 15), (3, 7, 15), (3, 15, 16), (3, 16, 17), (3, 17, 18), (3, 18, 19), (3, 19, 20))
NAME_BEST_1ST_ORDER = "best_1st_order"

# ID = 210  / FOUND AT = 35753 
BEST_2ND_ORDER = ((-1, 1, 3, 32, 32), (0, 3, 128, 3, 3), (0, 128, 128, 3, 3), (0, 128, 128, 4, 4), (0, 128, 128, 3, 3), (0, 128, 256, 5, 5, 2), (0, 384, 128, 3, 3, 2), (0, 128, 128, 3, 3), (0, 128, 512, 3, 3, 2), (0, 128, 128, 2, 2), (0, 256, 256, 3, 3), (0, 256, 256, 5, 5), (0, 512, 256, 3, 3), (0, 256, 256, 3, 3), (0, 768, 512, 4, 4, 2), (0, 512, 512, 4, 4), (0, 512, 512, 2, 2), (0, 512, 512, 3, 3), (0, 1024, 512, 2, 2), (0, 512, 256, 8, 8), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 0, 5), (3, 4, 5), (3, 5, 6), (3, 6, 7), (3, 5, 8), (3, 4, 9), (3, 8, 10), (3, 9, 10), (3, 10, 11), (3, 4, 11), (3, 11, 12), (3, 12, 13), (3, 7, 13), (3, 7, 14), (3, 14, 15), (3, 15, 16), (3, 13, 17), (3, 16, 17), (3, 17, 18), (3, 18, 19), (3, 19, 20))
NAME_BEST_2ND_ORDER = "best_2nd_order"

def dropout_function(base_p, total_layers, index_layer, isPool=False):

    value = 0
    if index_layer != 0 and isPool == False:
        value = base_p

    if index_layer == total_layers - 2:
        value = base_p

    #print("conv2d: ", index_layer, " - dropout: ", value)

    return value

def pcos(x):
    if x>np.pi:
        x-np.pi
    return np.cos(x)

def Alaising(M,m,ep):
    M=10**(-M)
    m=10**(-m)
    return [ m+1/2*(M-m)*(1+pcos(t/ep*np.pi))
             for t in range(0,ep)]

def createNetwork(dna, settings):

    return nw.Network(adn=dna, cuda_flag=settings.cuda,
                                    momentum=settings.momentum, weight_decay=settings.weight_decay,
                                    enable_activation=settings.enable_activation,
                                    enable_track_stats=settings.enable_track_stats, dropout_value=settings.dropout_value,
                                    dropout_function=settings.dropout_function, enable_last_activation=settings.enable_last_activation,
                                    version=settings.version, eps_batchnorm=settings.eps_batchorm)

def trainNetwork(network, fileManager, memoryManager, settings, name):

    print("### TRAINING ", name, " ###")
    network.training_custom_dt(settings.dataGen, settings.joined_dt_array, settings.ricap)
    network.generate_accuracy(settings.dataGen)
    current_accuracy = network.get_accuracy()
    print("current accuracy=", current_accuracy)
    memoryManager.deleteNetwork(network)
    fileManager.appendFile("### Model: "+name+" - Acc: "+str(current_accuracy))


if __name__ == '__main__':

    DATA_SOURCE = 'cifar'

    settings = ExperimentSettings.ExperimentSettings()

    settings.cuda = True

    augSettings = AugmentationSettings.AugmentationSettings()

    dict_transformations = {
        augSettings.baseline_customRandomCrop : True,
        augSettings.randomHorizontalFlip : True,
        augSettings.randomErase_1 : True,
    }

    transform_compose = augSettings.generateTransformCompose(dict_transformations, False)
    # DIRECTIONS VERSION
    settings.version = directions_version.CONVEX_VERSION
    # NUM OF THREADS
    THREADS = int(input("Enter threads: "))

    # BATCH SIZE
    settings.batch_size = 64

    e =  50000 / settings.batch_size
    e = math.ceil(e)
    print("e = ", e)
    settings.iteration_per_epoch = e

    # Training Parameters
    settings.eps_batchorm = 0.001
    settings.dropout_value = 0.05
    settings.weight_decay = 0.0005
    settings.momentum = 0.9
    settings.enable_activation = True
    settings.enable_last_activation = True
    settings.allow_interupts = False
    settings.enable_track_stats = True
    settings.dropout_function = dropout_function
    JOINED_ITER = 100*e
    settings.joined_dt_array = Alaising(1.2,99,JOINED_ITER)

    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
    dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE, threads=THREADS, dataAugmentation=True)
    dataGen = dataCreator.returnParam()
    
    settings.dataGen = dataGen
    settings.ricap = Augmentation.Ricap(beta=0.3)

    memoryManager = MemoryManager.MemoryManager()

    # 1st order

    network_1st_5L = createNetwork(dna=DNA_product_4_1st_5L, settings=settings)
    trainNetwork(network=network_1st_5L, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_DNA_product_4_1st_5L)

    network_1st_6L = createNetwork(dna=DNA_product_4_1st_6L, settings=settings)
    trainNetwork(network=network_1st_6L, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_DNA_product_4_1st_6L)

    network_1st_7L = createNetwork(dna=DNA_product_4_1st_7L, settings=settings)
    trainNetwork(network=network_1st_7L, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_DNA_product_4_1st_7L)

    network_1st_8L = createNetwork(dna=DNA_product_4_1st_8L, settings=settings)
    trainNetwork(network=network_1st_8L, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_DNA_product_4_1st_8L)

    network_1st_9L = createNetwork(dna=DNA_product_4_1st_9L, settings=settings)
    trainNetwork(network=network_1st_9L, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_DNA_product_4_1st_9L)



    # 2nd order

    network_2nd_5L = createNetwork(dna=DNA_product_4_2nd_5L, settings=settings)
    trainNetwork(network=network_2nd_5L, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_DNA_product_4_2nd_5L)

    network_2nd_6L = createNetwork(dna=DNA_product_4_2nd_6L, settings=settings)
    trainNetwork(network=network_2nd_6L, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_DNA_product_4_2nd_6L)

    network_2nd_7L = createNetwork(dna=DNA_product_4_2nd_7L, settings=settings)
    trainNetwork(network=network_2nd_7L, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_DNA_product_4_2nd_7L)

    network_2nd_8L = createNetwork(dna=DNA_product_4_2nd_8L, settings=settings)
    trainNetwork(network=network_2nd_8L, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_DNA_product_4_2nd_8L)

    network_2nd_9L = createNetwork(dna=DNA_product_4_2nd_9L, settings=settings)
    trainNetwork(network=network_2nd_9L, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_DNA_product_4_2nd_9L)

    

    # Best 1st and 2nd order

    network_best_1st = createNetwork(dna=BEST_1ST_ORDER, settings=settings)
    trainNetwork(network=network_best_1st, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_BEST_1ST_ORDER)

    network_best_2nd = createNetwork(dna=BEST_2ND_ORDER, settings=settings)
    trainNetwork(network=network_best_2nd, fileManager=fileManager, memoryManager=memoryManager, settings=settings, name=NAME_BEST_2ND_ORDER)
