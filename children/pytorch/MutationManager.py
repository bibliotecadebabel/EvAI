import const.versions as versions_mutation
import children.pytorch.MutateNetwork_Dendrites as mutation_dendrites
import children.pytorch.MutateNetwork_Dendrites_H as mutation_h
import children.pytorch.MutateNetwork_Dendrites_clone as mutation_clone
import children.pytorch.MutateNetwork_Dendrites_pool as mutation_pool
import children.pytorch.MutateNetwork_Dendrites_duplicate as mutation_duplicate
import torch

class MutationManager():

    def __init__(self, directions_version=versions_mutation.DENDRITES_VERSION):
        
        self.__mutation_function = None

        print("VERSION SELECTED: ", directions_version)
        
        if directions_version == versions_mutation.H_VERSION:
            self.__mutation_function = mutation_h

        elif directions_version == versions_mutation.DUPLICATE_VERSION:
            self.__mutation_function = mutation_duplicate

        elif directions_version == versions_mutation.POOL_VERSION:
            self.__mutation_function = mutation_pool

        elif directions_version == versions_mutation.CLONE_VERSION:
            self.__mutation_function = mutation_clone
        
        else:
            self.__mutation_function = mutation_dendrites

    def getMuation(self):
        
        return self.__mutation_function
    
    def executeMutation(self, network, newAdn):

        newNetwork = self.__mutation_function.executeMutation(network, newAdn)
        torch.cuda.empty_cache()
        
        return newNetwork

