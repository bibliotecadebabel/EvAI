from Logic.commands import CommandCreateDataGen
import utilities.ExperimentSettings as ExperimentSettings
import const.versions as directions_version
import const.file_names as FileNames
import numpy as np
import utilities.Augmentation as Augmentation_Utils
import math
import children.pytorch.NetworkDendrites as nw
import utilities.FileManager as FileManager
import utilities.CheckpointModel as CheckPoint
import json


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

if __name__ == '__main__':

    print("#### POST TRAINING ####")

    settings = ExperimentSettings.ExperimentSettings()
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

    # Training parameters
    
    INIT_ITER = 200*e
    settings.init_dt_array =  Alaising(1.2,99,INIT_ITER)
    settings.max_init_iter = 3
    settings.dropout_value = 0.05
    settings.weight_decay = 0.0005
    settings.momentum = 0.9
    settings.cuda = True
    settings.dropout_function = dropout_function_constant
    settings.eps_batchorm = 0.001
    settings.enable_activation = True
    settings.enable_last_activation = True
    settings.enable_track_stats = True
    settings.enable_augmentation = True
    settings.ricap = Augmentation_Utils.Ricap(beta=0.3)

    dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=settings.cuda)
    dataCreator.execute(compression=2, batchSize=settings.batch_size, source=DATA_SOURCE, threads=THREADS, dataAugmentation=settings.enable_augmentation)
    dataGen = dataCreator.returnParam()

    settings.dataGen = dataGen
  
    fileManager = FileManager.FileManager()
    fileManager.setFileName(FileNames.POST_TRAINING_ACC)
    fileManager.writeFile("")
    avg_factor = len(settings.init_dt_array) // 4

    checkpoint_list = []

    with open(FileNames.SAVED_MODELS) as file_models:
        line = file_models.readline()
        while line:
            json_object = json.loads(line)
            checkpoint = CheckPoint.CheckPointModel(alaiTime=int(json_object['alaiTime']), dna=json_object['dna'])
            checkpoint.formatDNA()
            checkpoint_list.append(checkpoint)
            line = file_models.readline()

    max_alai = 0
    dna = None
    for item in checkpoint_list:

        if item.alaiTime >= max_alai:
            max_alai = item.alaiTime
            dna = item.dna

    
    network = nw.Network(dna,cudaFlag=settings.cuda,
                momentum=settings.momentum,
                weight_decay=settings.weight_decay,
                enable_activation=settings.enable_activation,
                enable_track_stats=settings.enable_track_stats,
                dropout_value=settings.dropout_value,
                dropout_function=settings.dropout_function,
                enable_last_activation=settings.enable_last_activation,
                version=settings.version, eps_batchnorm=settings.eps_batchorm)

    print("Model: ", network.adn)
    fileManager.appendFile("MODEL: "+str(network.adn))
    print("Starting training: 3 iterations of 200 epochs (total epochs: 600)")
    for i in range(settings.max_init_iter):
        print("iteration: ", i+1)
        network.iterTraining(settings.dataGen, settings.init_dt_array, settings.ricap)
        network.generateEnergy(settings.dataGen)
        current_accuracy = network.getAcurracy()
        print("current accuracy=", current_accuracy)
            
        if settings.save_txt == True:
            loss = network.getAverageLoss(avg_factor)
            fileManager.appendFile("iter: "+str(i+1)+" - Acc: "+str(current_accuracy)+" - Loss: "+str(loss))
    