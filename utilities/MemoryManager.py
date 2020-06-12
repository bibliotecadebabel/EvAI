import torch
import utilities.NetworkStorage as NetworkStorage
import os
import time
import torch
class MemoryManager():

    def __init__(self):
        self.__dynamic_net_dict = {}
        self.__basepath = os.path.join("saved_models","temp_models")

        if os.path.exists(self.__basepath) == False:
            os.makedirs(self.__basepath) 
    
    def saveNetwork(self, network):

        cuda = network.cudaFlag
        current_time = str(time.time())
        file_name = "tempmodel_"+current_time
        path = os.path.join(self.__basepath, file_name)
        network.saveModel(path)

        old_filename = self.getFileNameByKey(network.adn)

        if old_filename is not None:
            
            self.removeKey(network.adn)
            delete_path = os.path.join(self.__basepath, old_filename)

            if os.path.exists(delete_path):
                os.remove(delete_path)

        self.__dynamic_net_dict[network.adn] = file_name

        del network

        if cuda == True:
            torch.cuda.empty_cache()

    
    def loadNetwork(self, adn, settings):

        file_name = self.getFileNameByKey(adn)
        network_loaded = None
        
        if file_name == None:
            print("No network saved with adn: ", adn)
        else:
            path = os.path.join(self.__basepath, file_name)
            network_loaded = NetworkStorage.loadNetwork(fileName=file_name, settings=settings, path=path)
        
        return network_loaded
    
    def getFileNameByKey(self, adn):
        
        file_name = self.__dynamic_net_dict.get(adn)
        return file_name

    def removeKey(self, adn):

        file_name = self.__dynamic_net_dict.get(adn)

        if file_name is not None:
            del self.__dynamic_net_dict[adn]






        

