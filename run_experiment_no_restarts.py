import children.pytorch.NetworkDendrites as nw
import os
from DAO import GeneratorFromCIFAR
from DAO.database.dao import TestDAO, TestModelDAO
from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising, Damped_Alaising)

CUDA = True

TEST_NAME = "experiment_batchnorm_ver2"
TEST_DAO = TestDAO.TestDAO(db='database.db')
MODEL_DAO = TestModelDAO.TestModelDAO(db='database.db')

EPOCHS_1 = 5
EPOCHS_2 = 20

WEIGHT_DECAY = 0.00001
MOMENTUM = 0.9

BASE_DT = 0.0001
MIN_DT = 0.00001

BASE_DT_2 = 0.0001
MIN_DT_2 = 0.000001

BATCH_SIZE = 128


FOLDER = "cifar"


dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  BATCH_SIZE)
dataGen.dataConv2d()


ADN =  ((-1, 1, 3, 32, 32), (0, 3, 60, 3, 3), (0, 60, 60, 3, 3), (0, 60, 60, 3, 3), 
        (0, 183, 60, 3, 3), (0, 120, 60, 5, 5), (0, 183, 60, 3, 3), (0, 60, 60, 3, 3), 
        (0, 180, 60, 3, 3), (0, 60, 60, 3, 3), (0, 180, 30, 4, 4), (0, 30, 60, 3, 3), 
        (0, 150, 60, 4, 4), (0, 150, 60, 4, 4), (0, 333, 60, 32, 32), (1, 60, 10), (2,), 
        (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, -1, 3), (3, 0, 3), (3, 1, 3), (3, 2, 3), (3, 3, 4),
         (3, 2, 4), (3, 3, 5), (3, -1, 5), (3, 4, 5), (3, 1, 5), (3, 5, 6), (3, 5, 7), (3, 4, 7), 
         (3, 6, 7), (3, 7, 8), (3, 5, 9), (3, 7, 9), (3, 8, 9), (3, 9, 10), (3, 9, 11), (3, 10, 11),
          (3, 7, 11), (3, 11, 12), (3, 9, 12), (3, 10, 12), (3, 5, 13), (3, 3, 13), (3, -1, 13), (3, 9, 13), 
          (3, 11, 13), (3, 12, 13), (3, 10, 13), (3, 13, 14), (3, 14, 15))



network = nw.Network(adn=ADN, cudaFlag=CUDA, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
test_id = TEST_DAO.insert(testName=TEST_NAME, periodSave=1, dt=BASE_DT, total=(EPOCHS_1+EPOCHS_2), periodCenter=0)

print("STARTING TRAINING")
network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=BASE_DT, min_dt=MIN_DT, 
                                    epochs=EPOCHS_1, restart_dt=EPOCHS_1, show_accuarcy=False)

network.generateEnergy(dataGen=dataGen)
print("ACCURACY=", network.getAcurracy())
FILE_NAME = str(test_id)+"_"+TEST_NAME+"_"+str(5)
PATH_SAVE = os.path.join("saved_models", FOLDER, FILE_NAME)
MODEL_DAO.insert(idTest=test_id, dna=str(ADN), iteration=5, 
                        fileName=FILE_NAME, model_weight=network.getAcurracy())
network.saveModel(path=PATH_SAVE)

network.TrainingCosineLR_Restarts(dataGenerator=dataGen, max_dt=BASE_DT_2, min_dt=MIN_DT_2, 
                                    epochs=EPOCHS_2, restart_dt=EPOCHS_2, show_accuarcy=True)
print("FINISH TRAINING")

network.generateEnergy(dataGen)
print("FINAL ACCURACY")
print(network.getAcurracy())

FILE_NAME = str(test_id)+"_"+TEST_NAME+"_"+str(25)
PATH_SAVE = os.path.join("saved_models", FOLDER, FILE_NAME)
MODEL_DAO.insert(idTest=test_id, dna=str(ADN), iteration=25, 
                        fileName=FILE_NAME, model_weight=network.getAcurracy())
network.saveModel(path=PATH_SAVE)

