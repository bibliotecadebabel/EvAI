import torch
import utilities.NetworkStorage as NetworkStorage
import os
import time
import torch
import gc

class MemoryManager():

    def __init__(self):
        self.__dynamic_net_dict = {}
        self.__basepath = os.path.join("saved_models","temp_models")

        if os.path.exists(self.__basepath) == False:
            os.makedirs(self.__basepath) 
        
        self.clearTempModels()
    
    def clearTempModels(self):

        for f in os.listdir(self.__basepath):
            
            path = os.path.join(self.__basepath, f)
            os.remove(path)

    def saveTempNetwork(self, network):

        current_time = str(time.time())
        file_name = "tempmodel_"+current_time
        path = os.path.join(self.__basepath, file_name)
        network.saveModel(path)

        old_filename = self.getFileNameByKey(network.adn)

        if old_filename is not None:
            self.removeNetwork(network.adn, deleteFile=True)

        self.__dynamic_net_dict[network.adn] = file_name

        self.deleteNetwork(network)

    
    def loadTempNetwork(self, adn, settings):

        file_name = self.getFileNameByKey(adn)
        network_loaded = None
        
        if file_name == None:
            print("No network saved with adn: ", adn)
        else:
            path = os.path.join(self.__basepath, file_name)
            network_loaded = NetworkStorage.loadNetwork(fileName=file_name, settings=settings, path=path)

        gc.collect()
        if network_loaded.cudaFlag == True:
            torch.cuda.empty_cache()

        return network_loaded
    
    def getFileNameByKey(self, adn):
        
        file_name = self.__dynamic_net_dict.get(adn)
        return file_name

    def removeNetwork(self, adn, deleteFile=False):

        file_name = self.__dynamic_net_dict.get(adn)

        if file_name is not None:
            del self.__dynamic_net_dict[adn]

            if deleteFile == True:
                delete_path = os.path.join(self.__basepath, file_name)

                if os.path.exists(delete_path):
                    os.remove(delete_path)
   
    def deleteNetwork(self, network):

        cuda = network.cudaFlag
        network.deleteParameters()
        del network

        gc.collect()
        if cuda == True:
            torch.cuda.empty_cache()









        

