import const.versions as mutation_version
import mutations.versions.mutation_h as mutation_h
import mutations.versions.mutation_convex as mutation_convex
import mutations.versions.mutation_pool as mutation_pool
import mutations.versions.mutation_pool_duplicate as mutation_pool_duplicate
import torch

class MutationManager():

    def __init__(self, directions_version):
        
        self.__mutation_function = None

        print("mutation version: ", directions_version)
        
        if directions_version == mutation_version.H_VERSION:
            self.__mutation_function = mutation_h

        elif directions_version == mutation_version.POOL_VERSION:
            self.__mutation_function = mutation_pool
        
        elif directions_version == mutation_version.POOL_DUPLICATE_VERSION:
            self.__mutation_function = mutation_pool_duplicate
        
        elif directions_version == mutation_version.CONVEX_VERSION:
            self.__mutation_function = mutation_convex

        else:
            self.__mutation_function = mutation_pool

    def get_mutation_function(self):
        
        return self.__mutation_function
    
    def execute_mutation(self, network, new_dna):

        newNetwork = self.__mutation_function.execute_mutation(network, new_dna)
        torch.cuda.empty_cache()
        
        return newNetwork

