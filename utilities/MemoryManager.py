import torch
import utilities.NetworkStorage as NetworkStorage
import os
import time
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
        network.save_model(path)

        old_filename = self.getFileNameByKey(network.dna)

        if old_filename is not None:
            self.removeNetwork(network.dna, deleteFile=True)

        self.__dynamic_net_dict[network.dna] = file_name

        self.deleteNetwork(network)

    
    def loadTempNetwork(self, dna, settings):

        file_name = self.getFileNameByKey(dna)
        network_loaded = None
        
        if file_name == None:
            print("No network saved with dna: ", dna)
        else:
            path = os.path.join(self.__basepath, file_name)
            network_loaded = NetworkStorage.load_network(fileName=file_name, settings=settings, path=path)

        gc.collect()
        if network_loaded.cuda_flag == True:
            torch.cuda.empty_cache()

        return network_loaded
    
    def getFileNameByKey(self, dna):
        
        file_name = self.__dynamic_net_dict.get(dna)
        return file_name

    def removeNetwork(self, dna, deleteFile=False):

        file_name = self.__dynamic_net_dict.get(dna)

        if file_name is not None:
            del self.__dynamic_net_dict[dna]

            if deleteFile == True:
                delete_path = os.path.join(self.__basepath, file_name)

                if os.path.exists(delete_path):
                    os.remove(delete_path)
   
    def deleteNetwork(self, network):

        cuda = network.cuda_flag
        network.delete_parameters()
        del network

        gc.collect()
        if cuda == True:
            torch.cuda.empty_cache()









        

