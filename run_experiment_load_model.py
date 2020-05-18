import children.pytorch.NetworkDendrites as nw
import os
from DAO import GeneratorFromCIFAR
from DAO.database.dao import TestDAO, TestModelDAO

CUDA = True

TEST_NAME = "experiment_stored_model"
TEST_DAO = TestDAO.TestDAO()
MODEL_DAO = TestModelDAO.TestModelDAO()

BASE_DT = 0.1
MIN_DT = 0.00001
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

EPOCHS = 60
BATCH_SIZE = 128
RESTAR_EPOCH_PERIOD = 30

FOLDER = "cifar"
MODEL_NAME = "4_cifar_test_restart-clone_2_model_25"
STORED_MODEL_NAME = TEST_NAME+"_"+str(EPOCHS)

PATH_LOAD = os.path.join("saved_models", FOLDER, MODEL_NAME)


dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  BATCH_SIZE)
dataGen.dataConv2d()

ADN = ((-1, 1, 3, 32, 32), (0, 3, 60, 3, 3), (0, 60, 60, 3, 3), 
        (0, 60, 60, 3, 3), (0, 183, 60, 3, 3), (0, 60, 60, 3, 3), 
        (0, 123, 60, 3, 3), (0, 60, 60, 3, 3), (0, 120, 30, 4, 4), 
        (0, 30, 60, 3, 3), (0, 90, 60, 3, 3), (0, 150, 60, 4, 4), 
        (0, 273, 60, 32, 32), (1, 60, 10), (2,), (3, -1, 0), (3, 0, 1), 
        (3, 1, 2), (3, -1, 3), (3, 0, 3), (3, 1, 3), (3, 2, 3), (3, 3, 4), 
        (3, 3, 5), (3, -1, 5), (3, 4, 5), (3, 5, 6), (3, 5, 7), (3, 6, 7), 
        (3, 7, 8), (3, 7, 9), (3, 8, 9), (3, 9, 10), (3, 7, 10), (3, 8, 10), 
        (3, 5, 11), (3, 3, 11), (3, -1, 11), (3, 7, 11), (3, 9, 11), (3, 10, 11), (3, 11, 12), (3, 12, 13))


network = nw.Network(adn=ADN, cudaFlag=CUDA, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
network.loadModel(path=PATH_LOAD)

network.generateEnergy(dataGen)

print("CURRENT ACCURACY")
print(network.getAcurracy())

print("STARTING TRAINING")
test_id = TEST_DAO.insert(testName=TEST_NAME, periodSave=EPOCHS, dt=BASE_DT, total=EPOCHS, periodCenter=0)
network.TrainingCosineLR_Restarts(dataGenerator=dataGen, base_dt=BASE_DT,epochs=EPOCHS,etamin=MIN_DT, period_restart=RESTAR_EPOCH_PERIOD)
print("FINISH TRAINING")

network.generateEnergy(dataGen)
print("NEW ACCURACY")
print(network.getAcurracy())

FILE_NAME = str(test_id)+"_"+STORED_MODEL_NAME

PATH_SAVE = os.path.join("saved_models", FOLDER, FILE_NAME)
MODEL_DAO.insert(idTest=test_id, dna=str(ADN), iteration=EPOCHS, fileName=STORED_MODEL_NAME, model_weight=network.getAcurracy())
network.saveModel(path=PATH_SAVE)

