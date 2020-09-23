import children.pytorch.network_dendrites as nw
from Logic.commands import CommandCreateDataGen
from DAO.database.dao import TestDAO, TestResultDAO, TestModelDAO
import numpy as np
import utilities.Augmentation as Augmentation
import utilities.AugmentationSettings as AugmentationSettings
import utilities.ExperimentSettings as ExperimentSettings
import const.versions as directions_version
import utilities.MemoryManager as MemoryManager
import math

modelDAO = TestModelDAO.TestModelDAO()

# found at 1200 epochs (dont train)
BEST_LSTM_1 = ((-1, 1, 3, 32, 32), (0, 3, 64, 4, 4), (0, 64, 128, 3, 3, 2), (0, 128, 258, 3, 3, 2), (0, 258, 259, 5, 5), (1, 259, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5))

# Train for 100 epochs
BEST_product_1 = ((-1, 1, 3, 32, 32), (0, 3, 52, 4, 4), (0, 52, 118, 3, 3, 2), (0, 118, 252, 4, 4, 2), (0, 252, 259, 4, 4), (1, 259, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5))
ID_BEST_product_1 = 1016

# Train for 100 epochs
DNA_product_2_5L = ((-1, 1, 3, 32, 32), (0, 3, 64, 2, 2), (0, 64, 128, 3, 3, 2), (0, 128, 128, 3, 3), (0, 256, 256, 3, 3, 2), (0, 256, 256, 5, 5), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 1, 3), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6))
ID_DNA_product_2_5L = 1320

# Train for 100 epochs
DNA_product_2_6L = ((-1, 1, 3, 32, 32), (0, 3, 64, 2, 2), (0, 64, 128, 3, 3, 2), (0, 128, 64, 2, 2), (0, 192, 128, 4, 4), (0, 256, 256, 3, 3, 2), (0, 256, 256, 5, 5), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 1, 3), (3, 2, 3), (3, 1, 4), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 6, 7))
ID_DNA_product_2_6L = 1324

# Train for 100 epochs
DNA_product_3_5L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 67, 64, 2, 2), (0, 128, 128, 3, 3, 2), (0, 128, 256, 3, 3, 2), (0, 256, 256, 5, 5), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, -1, 1), (3, 0, 2), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5), (3, 5, 6))
ID_DNA_product_3_5L = 1582

# Train for 100 epochs
DNA_product_3_6L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 128, 2, 2), (0, 192, 128, 3, 3, 2), (0, 192, 128, 2, 2, 2), (0, 256, 256, 3, 3, 2), (0, 256, 256, 5, 5), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 0, 2), (3, 1, 2), (3, 2, 3), (3, 0, 3), (3, 2, 4), (3, 3, 4), (3, 4, 5), (3, 5, 6), (3, 6, 7))
ID_DNA_product_3_6L = 1585

# Train for 100 epochs
DNA_product_3_7L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 128, 4, 4), (0, 192, 32, 4, 4), (0, 96, 128, 3, 3, 2), (0, 160, 256, 2, 2, 2), (0, 384, 256, 3, 3, 2), (0, 256, 256, 5, 5), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 0, 2), (3, 1, 2), (3, 0, 3), (3, 2, 3), (3, 3, 4), (3, 2, 4), (3, 3, 5), (3, 4, 5), (3, 5, 6), (3, 6, 7), (3, 7, 8))
ID_DNA_product_3_7L = 1602

# Train for 100 epochs
DNA_product_3_8L = ((-1, 1, 3, 32, 32), (0, 3, 64, 3, 3), (0, 64, 64, 3, 3, 2), (0, 128, 128, 4, 4), (0, 192, 32, 5, 5), (0, 96, 128, 3, 3, 2), (0, 192, 256, 2, 2, 2), (0, 384, 256, 3, 3, 2), (0, 256, 256, 5, 5), (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 0, 2), (3, 1, 2), (3, 0, 3), (3, 2, 3), (3, 0, 4), (3, 3, 4), (3, 4, 5), (3, 0, 5), (3, 4, 6), (3, 5, 6), (3, 6, 7), (3, 7, 8), (3, 8, 9))
ID_DNA_product_3_8L = 1608

# Train for 100 epochs
BEST_product_2 = ((-1, 1, 3, 32, 32), (0, 3, 4, 6, 6), (0, 7, 4, 2, 2), (0, 7, 64, 3, 3), (0, 67, 4, 4, 4), (0, 7, 64, 3, 3), (0, 64, 4, 3, 3), (0, 68, 8, 2, 2), (0, 72, 8, 8, 8), (0, 72, 256, 2, 2), (0, 320, 128, 3, 3, 2), (0, 128, 4, 5, 5), (0, 132, 128, 2, 2), (0, 256, 8, 3, 3), (0, 136, 16, 3, 3), (0, 144, 256, 5, 5), (0, 384, 16, 2, 2), (0, 144, 4, 3, 3), (0, 132, 256, 4, 4, 2), (0, 256, 128, 4, 4), (1, 128, 10), (2,), (3, -1, 0), (3, -1, 1), (3, 0, 1), (3, -1, 2), (3, 1, 2), (3, -1, 3), (3, 2, 3), (3, -1, 4), (3, 3, 4), (3, 4, 5), (3, 4, 6), (3, 5, 6), (3, 4, 7), (3, 6, 7), (3, 4, 8), (3, 7, 8), (3, 4, 9), (3, 8, 9), (3, 9, 10), (3, 9, 11), (3, 10, 11), (3, 9, 12), (3, 11, 12), (3, 9, 13), (3, 12, 13), (3, 9, 14), (3, 13, 14), (3, 9, 15), (3, 14, 15), (3, 9, 16), (3, 15, 16), (3, 9, 17), (3, 16, 17), (3, 17, 18), (3, 18, 19), (3, 19, 20))
ID_BEST_product_2 = 1526

# Train for 100 epochs
BEST_product_3 = ((-1, 1, 3, 32, 32), (0, 3, 16, 3, 3), (0, 19, 32, 2, 2), (0, 35, 32, 3, 3), (0, 35, 4, 4, 4), (0, 7, 8, 5, 5), (0, 11, 4, 5, 5), (0, 7, 4, 2, 2), (0, 7, 32, 2, 2), (0, 35, 16, 9, 9), (0, 19, 64, 3, 3), (0, 80, 4, 3, 3), (0, 68, 16, 6, 6), (0, 80, 64, 3, 3, 2), (0, 128, 32, 5, 5), (0, 96, 256, 2, 2), (0, 320, 128, 3, 3, 2), (0, 384, 64, 5, 5, 2), (0, 192, 256, 3, 3, 2), (0, 256, 256, 5, 5), (1, 256, 10), (2,), (3, -1, 0), (3, -1, 1), (3, 0, 1), (3, -1, 2), (3, 1, 2), (3, -1, 3), (3, 2, 3), (3, -1, 4), (3, 3, 4), (3, -1, 5), (3, 4, 5), (3, -1, 6), (3, 5, 6), (3, -1, 7), (3, 6, 7), (3, -1, 8), (3, 7, 8), (3, -1, 9), (3, 8, 9), (3, 9, 10), (3, 8, 10), (3, 10, 11), (3, 9, 11), (3, 9, 12), (3, 11, 12), (3, 9, 13), (3, 12, 13), (3, 9, 14), (3, 13, 14), (3, 9, 15), (3, 14, 15), (3, 15, 16), (3, 14, 16), (3, 15, 17), (3, 16, 17), (3, 17, 18), (3, 18, 19), (3, 19, 20))
ID_BEST_product_3 = 1751

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

def trainNetwork(network, id, dao, memoryManager, settings, name):

    print("### TRAINING ", name, " ###")
    network.training_custom_dt(settings.dataGen, settings.joined_dt_array, settings.ricap)
    network.generate_accuracy(settings.dataGen)
    current_accuracy = network.get_accuracy()
    print("current accuracy=", current_accuracy)
    memoryManager.deleteNetwork(network)
    dao.updateAcc(acc=current_accuracy, idModel=id)


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
    settings.version = directions_version.POOL_VERSION
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

    # product_1
    product_1_network = createNetwork(dna=BEST_product_1, settings=settings)
    trainNetwork(network=product_1_network, id=ID_BEST_product_1, dao=modelDAO, memoryManager=memoryManager, settings=settings, name="product_1_network")



    # product_2
    product_2_5l_network = createNetwork(dna=DNA_product_2_5L, settings=settings)
    trainNetwork(network=product_2_5l_network, id=ID_DNA_product_2_5L, dao=modelDAO, memoryManager=memoryManager, settings=settings, name="product_2_5l_network")

    product_2_6l_network = createNetwork(dna=DNA_product_2_6L, settings=settings)
    trainNetwork(network=product_2_6l_network, id=ID_DNA_product_2_6L, dao=modelDAO, memoryManager=memoryManager, settings=settings, name="product_2_6l_network")



    # product_3
    product_3_5l_network = createNetwork(dna=DNA_product_3_5L, settings=settings)
    trainNetwork(network=product_3_5l_network, id=ID_DNA_product_3_5L, dao=modelDAO, memoryManager=memoryManager, settings=settings, name="product_3_5l_network")

    product_3_6l_network = createNetwork(dna=DNA_product_3_6L, settings=settings)
    trainNetwork(network=product_3_6l_network, id=ID_DNA_product_3_6L, dao=modelDAO, memoryManager=memoryManager, settings=settings, name="product_3_6l_network")

    product_3_7l_network = createNetwork(dna=DNA_product_3_7L, settings=settings)
    trainNetwork(network=product_3_7l_network, id=ID_DNA_product_3_7L, dao=modelDAO, memoryManager=memoryManager, settings=settings, name="product_3_7l_network")

    product_3_8l_network = createNetwork(dna=DNA_product_3_8L, settings=settings)
    trainNetwork(network=product_3_8l_network, id=ID_DNA_product_3_8L, dao=modelDAO, memoryManager=memoryManager, settings=settings, name="product_3_8l_network")




    # best of product 2 & 3
    product_2_best_network = createNetwork(dna=BEST_product_2, settings=settings)
    trainNetwork(network=product_2_best_network, id=ID_BEST_product_2, dao=modelDAO, memoryManager=memoryManager, settings=settings, name="product_2_best_network")

    product_3_best_network = createNetwork(dna=BEST_product_3, settings=settings)
    trainNetwork(network=product_3_best_network, id=ID_BEST_product_3, dao=modelDAO, memoryManager=memoryManager, settings=settings, name="product_3_best_network")