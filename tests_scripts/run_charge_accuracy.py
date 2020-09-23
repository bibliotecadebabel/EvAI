import DAO.database.dao.TestModelDAO as TestModelDAO
import random
import os
import utilities.NetworkStorage as NetworkStorage
import utilities.MemoryManager as MemoryManager
from TestNetwork.commands import CommandCreateDataGen
import TestNetwork.ExperimentSettings as ExperimentSettings

testModelDAO = TestModelDAO.TestModelDAO()
memoryManager = MemoryManager.MemoryManager()

settings = ExperimentSettings.ExperimentSettings()
settings.version = "convex"
settings.batch_size = 64
settings.cuda = True
settings.eps_batchorm = 0.001
settings.dropout_value = 0.05
settings.weight_decay = 0.0005
settings.momentum = 0.9
settings.enable_activation = True
settings.enable_last_activation = True
settings.enable_track_stats = True


test_id = int(input("Enter id test: "))
max_alai_time = int(input("Enter max alai time: "))

test_models = testModelDAO.findByLimitAlai(idTest=test_id, limit_alai_time=max_alai_time)

dataCreator = CommandCreateDataGen.CommandCreateDataGen(cuda=True)
dataCreator.execute(compression=2, batchSize=settings.batch_size, source="cifar", threads=2, dataAugmentation=True)
dataGen = dataCreator.returnParam()
    
for model in test_models:
    print("charging model: ", model.model_name)
    path = os.path.join("saved_models","product_database", model.model_name)
    network = NetworkStorage.loadNetwork(fileName=None, settings=settings, path=path)
    network.generate_accuracy(dataGen)
    acc = network.get_accuracy()
    memoryManager.deleteNetwork(network)
    print(path)
    print("Acc: ", acc)
    testModelDAO.updateAcc(acc=acc, idModel=model.id)
