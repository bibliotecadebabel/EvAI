import children.pytorch.NetworkDendrites as nw
import os
from DAO import GeneratorFromCIFAR
from DAO.database.dao import TestDAO, TestModelDAO
from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising, Damped_Alaising)

CUDA = True

TEST_NAME = "experiment_aliasing"
TEST_DAO = TestDAO.TestDAO(db='database.db')
MODEL_DAO = TestModelDAO.TestModelDAO(db='database.db')

WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
EPOCHS = 200
BATCH_SIZE = 128
SHOW_ACCURACY_PERIOD = 5
MAX_ITER = EPOCHS * ((50000 // BATCH_SIZE) + 1) 

FOLDER = "cifar"
STORED_MODEL_NAME = TEST_NAME+"_"+str(EPOCHS)


dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  BATCH_SIZE)
dataGen.dataConv2d()

ADN = ((-1, 1, 3, 32, 32), (0, 3, 60, 3, 3), (0, 60, 60, 3, 3), (0, 60, 60, 3, 3), 
        (0, 183, 60, 3, 3), (0, 120, 60, 5, 5), (0, 183, 60, 3, 3), (0, 60, 60, 3, 3), 
        (0, 180, 60, 3, 3), (0, 60, 60, 3, 3), (0, 180, 30, 4, 4), (0, 30, 60, 3, 3), 
        (0, 150, 60, 4, 4), (0, 150, 60, 4, 4), (0, 333, 60, 32, 32), (1, 60, 10), (2,), 
        (3, -1, 0), (3, 0, 1), (3, 1, 2), (3, -1, 3), (3, 0, 3), (3, 1, 3), (3, 2, 3), (3, 3, 4),
         (3, 2, 4), (3, 3, 5), (3, -1, 5), (3, 4, 5), (3, 1, 5), (3, 5, 6), (3, 5, 7), (3, 4, 7), 
         (3, 6, 7), (3, 7, 8), (3, 5, 9), (3, 7, 9), (3, 8, 9), (3, 9, 10), (3, 9, 11), (3, 10, 11),
          (3, 7, 11), (3, 11, 12), (3, 9, 12), (3, 10, 12), (3, 5, 13), (3, 3, 13), (3, -1, 13), (3, 9, 13), 
          (3, 11, 13), (3, 12, 13), (3, 10, 13), (3, 13, 14), (3, 14, 15))


Alai= Damped_Alaising(Max_iter=MAX_ITER)
print("max iter=", Alai.Max_iter)

network = nw.Network(adn=ADN, cudaFlag=CUDA, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

print("STARTING TRAINING")
test_id = TEST_DAO.insert(testName=TEST_NAME, periodSave=EPOCHS, dt=Alai.current_max, total=EPOCHS, periodCenter=0)
network.TrainingALaising(dataGenerator=dataGen, epochs=EPOCHS, alaising_object=Alai, period_show_accuracy=SHOW_ACCURACY_PERIOD)
print("FINISH TRAINING")

network.generateEnergy(dataGen)
print("NEW ACCURACY")
print(network.getAcurracy())

FILE_NAME = str(test_id)+"_"+STORED_MODEL_NAME

PATH_SAVE = os.path.join("saved_models", FOLDER, FILE_NAME)
MODEL_DAO.insert(idTest=test_id, dna=str(ADN), iteration=EPOCHS, 
                        fileName=STORED_MODEL_NAME, model_weight=network.getAcurracy())
network.saveModel(path=PATH_SAVE)

