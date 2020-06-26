from TestNetwork.commands import CommandCreateDataGen, CommandGetAllTest
from TestNetwork.commands import  CommandExperimentCifar_Shuffle_generations as CommandExperimentCifar_Restarts
import TestNetwork.ExperimentSettings as ExperimentSettings
import TestNetwork.AugmentationSettings as AugmentationSettings
import const.versions as directions_version
import const.training_type as TrainingType
import numpy as np
import test_DNAs as DNAs
import utilities.Augmentation as Augmentation_Utils
import math
import children.pytorch.NetworkDendrites as nw
import utilities.FileManager as FileManager
import DAO.database.dao.TestModelDAO as TestModelDAO
import os
import utilities.NetworkStorage as NetworkStorage
import children.pytorch.MutationManager as MutationManager
import utilities.MemoryManager as MemoryManager

###### EXPERIMENT SETTINGS ######

def dropout_function_constant(base_p, total_layers, index_layer, isPool=False):

    value = 0
    if index_layer != 0 and isPool == False:
        value = base_p

    if index_layer == total_layers - 2:
        value = base_p

    print("conv2d: ", index_layer, " - dropout: ", value)

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

def saveModel(test, network, iteration):

    fileName = str(test.id)+"_post-training-model_alai35753_"+str(iteration)
    final_path = os.path.join("saved_models","product_database", fileName)

    network.saveModel(final_path)
    
    testModelDao.insert(idTest=test.id, dna=str(network.adn), iteration=iteration, fileName=fileName,
                        model_weight=network.getAcurracy(), training_type=TrainingType.POST_TRAINING, current_alai_time=iteration)

    print("model saved with acc: ", network.getAcurracy())

if __name__ == '__main__':

    print("#### POST TRAINING ####")

    getTestsCommand = CommandGetAllTest.CommandGetAllTest()

    getTestsCommand.execute()

    test_list = getTestsCommand.getReturnParam()

    testModelDao = TestModelDAO.TestModelDAO()

    for test in test_list:
        print("id:", test.id,"|| name:", test.name, "|| dt_range:", test.dt, "-", test.dt_min,"|| batchsize:", test.batch_size, "|| date:", test.start_time)

    print("")  
    print("############ SELECT TEST ############")
    TEST_ID = int(input("Select test id: "))

    selected_test = None

    for test in test_list:
        
        if TEST_ID == test.id:
            selected_test = test
            break

    if selected_test == None:
        raise RuntimeError("Test id does not exist.")
    
    print("Selected test:", selected_test.name, "(id:",selected_test.id,")")

    settings = ExperimentSettings.ExperimentSettings()

    augSettings = AugmentationSettings.AugmentationSettings()

    dict_transformations = {
        augSettings.baseline_customRandomCrop : True,
        augSettings.randomHorizontalFlip : True,
        augSettings.randomErase_1 : True
    }

    transform_compose = augSettings.generateTransformCompose(dict_transformations, False)

    # DIRECTIONS VERSION
    settings.version = directions_version.CONVEX_VERSION
    # NUM OF THREADS
    THREADS = int(input("Threads: "))

    # BATCH SIZE
    settings.batch_size = int(input("batch size: "))
    e =  50000 / settings.batch_size
    e = math.ceil(e)
    print("e = ", e)

    # DATA SOURCE ('default' -> Pikachu, 'cifar' -> CIFAR)
    DATA_SOURCE = 'cifar'

    # CUDA parameter (true/false)
    settings.cuda = True
    
    # Training parameters
    INIT_ITER = 200*e
    settings.init_dt_array =  Alaising(1.2,99,INIT_ITER)
    settings.max_init_iter = int(input("Total Iterations of "+str(len(settings.init_dt_array)//e)+" epochs: "))
    settings.dropout_value = 0.05
    settings.weight_decay = 0.0005
    settings.momentum = 0.9
    settings.dropout_function = dropout_function_constant
    settings.eps_batchorm = 0.001
    
    # ENABLE_ACTIVATION, enable/disable relu
    ENABLE_ACTIVATION = int(input("Enable activation? (1 = yes, 0 = no): "))

    value = False
    if ENABLE_ACTIVATION == 1:
        value = True
    settings.enable_activation = value

    # ENABLE_LAST_ACTIVATION, enable/disable last layer relu
    ENABLE_LAST_ACTIVATION = int(input("Enable last layer activation? (1 = yes, 0 = no): "))

    value = False
    if ENABLE_LAST_ACTIVATION == 1:
        value = True
    settings.enable_last_activation = value

    # ENABLE_AUGMENTATION, enable/disable data augmentation
    ENABLE_AUGMENTATION = int(input("Enable Data augmentation? (1 = yes, 0 = no): "))

    value = False
    if ENABLE_AUGMENTATION == 1:
        value = True
    ENABLE_AUGMENTATION = value

    # ALLOW TRACK BATCHNORM
    ENABLE_TRACK = int(input("Enable tracking var/mean batchnorm? (1 = yes, 0 = no): "))
    value = False
    if ENABLE_TRACK == 1:
        value = True
    settings.enable_track_stats = value

    settings.ricap = Augmentation_Utils.Ricap(beta=0.3)

    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
    dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE, threads=THREADS, dataAugmentation=ENABLE_AUGMENTATION)
    dataGen = dataCreator.returnParam()

    settings.dataGen = dataGen
    settings.save_txt = True
    

    fileManager = FileManager.FileManager()
        
    if settings.save_txt == True:
        fileManager.setFileName("post_training_"+selected_test.name)
        fileManager.writeFile("")

    #path = os.path.join("saved_models","product_database", "13_post-training-model_alai35753_0")
    #network = NetworkStorage.loadNetwork(fileName=None, settings=settings, path=path)

    INIT_DNA_TESTING = ((-1, 1, 3, 32, 32), (0, 3, 128, 3, 3), (0, 128, 128, 3, 3), (0, 128, 128, 4, 4), 
                (0, 128, 128, 3, 3), (0, 128, 256, 5, 5, 2), (0, 384, 128, 3, 3, 2), (0, 128, 128, 3, 3), 
                (0, 128, 512, 3, 3, 2), (0, 128, 128, 2, 2), (0, 256, 256, 3, 3), (0, 256, 256, 5, 5), 
                (0, 512, 256, 3, 3), (0, 256, 256, 3, 3), (0, 768, 512, 4, 4, 2), (0, 512, 512, 4, 4), 
                (0, 512, 512, 2, 2), (0, 512, 512, 3, 3), (0, 1024, 256, 2, 2), (0, 256, 256, 8, 8), 
                (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 0, 5), (3, 4, 5), 
                (3, 5, 6), (3, 6, 7), (3, 5, 8), (3, 4, 9), (3, 8, 10), (3, 9, 10), (3, 10, 11), (3, 4, 11), 
                (3, 11, 12), (3, 12, 13), (3, 7, 13), (3, 7, 14), (3, 14, 15), (3, 15, 16), (3, 13, 17), 
                (3, 16, 17), (3, 17, 18), (3, 18, 19), (3, 19, 20))

    network = nw.Network(adn=INIT_DNA_TESTING, cudaFlag=settings.cuda,
                                    momentum=settings.momentum, weight_decay=settings.weight_decay,
                                    enable_activation=settings.enable_activation,
                                    enable_track_stats=settings.enable_track_stats, dropout_value=settings.dropout_value,
                                    dropout_function=settings.dropout_function, enable_last_activation=settings.enable_last_activation,
                                    version=settings.version, eps_batchnorm=settings.eps_batchorm)
    network.generateEnergy(dataGen)
    print("Loaded acc: ", network.getAcurracy())

    #mutationManager = MutationManager.MutationManager(directions_version="convex")
    #memoryManager = MemoryManager.MemoryManager()

    '''
    NEW_DNA = ((-1, 1, 3, 32, 32), (0, 3, 128, 3, 3), (0, 128, 128, 3, 3), (0, 128, 128, 4, 4), 
                (0, 128, 128, 3, 3), (0, 128, 256, 5, 5, 2), (0, 384, 128, 3, 3, 2), (0, 128, 128, 3, 3), 
                (0, 128, 512, 3, 3, 2), (0, 128, 128, 2, 2), (0, 256, 256, 3, 3), (0, 256, 256, 5, 5), 
                (0, 512, 256, 3, 3), (0, 256, 256, 3, 3), (0, 768, 512, 4, 4, 2), (0, 512, 512, 4, 4), 
                (0, 512, 512, 2, 2), (0, 512, 512, 3, 3), (0, 1024, 512, 2, 2), (0, 512, 256, 8, 8), 
                (1, 256, 10), (2,), (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 0, 5), (3, 4, 5), 
                (3, 5, 6), (3, 6, 7), (3, 5, 8), (3, 4, 9), (3, 8, 10), (3, 9, 10), (3, 10, 11), (3, 4, 11), 
                (3, 11, 12), (3, 12, 13), (3, 7, 13), (3, 7, 14), (3, 14, 15), (3, 15, 16), (3, 13, 17), 
                (3, 16, 17), (3, 17, 18), (3, 18, 19), (3, 19, 20))

    oldNetwork = network
    network = mutationManager.executeMutation(network, NEW_DNA)
    '''
    avg_factor = len(settings.init_dt_array) // 4

    #memoryManager.deleteNetwork(oldNetwork)

    #network.generateEnergy(settings.dataGen)
    #saveModel(test=selected_test, network=network, iteration=0)

    for i in range(settings.max_init_iter):
        print("iteration: ", i+1)
        network.iterTraining(settings.dataGen, settings.init_dt_array, settings.ricap)
        network.generateEnergy(settings.dataGen)
        current_accuracy = network.getAcurracy()
        print("current accuracy=", current_accuracy)
            
        if settings.save_txt == True:
            loss = network.getAverageLoss(avg_factor)
            fileManager.appendFile("iter: "+str(i+1)+" - Acc: "+str(current_accuracy)+" - Loss: "+str(loss))
    
    
    iteration = settings.max_init_iter * len(settings.init_dt_array)
    saveModel(test=selected_test, network=network, iteration=iteration)


