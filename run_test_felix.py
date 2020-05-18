import children.pytorch.NetworkDendrites as nw
import os
from DAO import GeneratorFromCIFAR
from DAO.database.dao import TestDAO, TestModelDAO

CUDA = True

TEST_NAME = "experiment_best_model"
TEST_DAO = TestDAO.TestDAO()
MODEL_DAO = TestModelDAO.TestModelDAO()

BASE_DT = 0.0005
MIN_DT = 0.000001
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

EPOCHS = 60
BATCH_SIZE = 128
RESTAR_EPOCH_PERIOD = 30
SHOW_ACCURACY_PERIOD = 2

FOLDER = "cifar"
STORED_MODEL_NAME = TEST_NAME+"_"+str(EPOCHS)



dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  BATCH_SIZE)
dataGen.dataConv2d()

ADN = ((-1,1,3,32,32),(0,3, 15, 3 , 3),(0,18, 15, 3,  3),(0,33, 15, 32, 32),(1, 15, 10),
             (2,),(3,-1,0),(3,0,1),(3,-1,1),(3,1,2),(3,0,2),(3,-1,2),(3,2,3),(3,3,4))

network = nw.Network(adn=ADN, cudaFlag=CUDA, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

print("STARTING TRAINING")
#test_id = TEST_DAO.insert(testName=TEST_NAME, periodSave=EPOCHS, dt=BASE_DT, total=EPOCHS, periodCenter=0)
network.TrainingCosineLR_Restarts(dataGenerator=dataGen, base_dt=BASE_DT,epochs=EPOCHS,
                                    etamin=MIN_DT, period_restart=RESTAR_EPOCH_PERIOD, 
                                    period_show_accuracy=SHOW_ACCURACY_PERIOD)
print("FINISH TRAINING")

network.generateEnergy(dataGen)
print("NEW ACCURACY")
print(network.getAcurracy())

#FILE_NAME = str(test_id)+"_"+STORED_MODEL_NAME

#PATH_SAVE = os.path.join("saved_models", FOLDER, FILE_NAME)
#MODEL_DAO.insert(idTest=test_id, dna=str(ADN), iteration=EPOCHS, 
#                        fileName=STORED_MODEL_NAME, model_weight=network.getAcurracy())
#network.saveModel(path=PATH_SAVE)

