import mutations.layers.conv2d.mutations as Conv2dMutations
import mutations.layers.linear.mutations as LinearMutations

class MutationsDictionary():

    def __init__(self):

        self.__mutations = {}
        
        self.__conv2d = {}
        self.__linear = {}
        
        self.__generate_conv2d_mutations()        
        self.__generate_linear_mutations()

        self.__generateLayerType()

    def __generateLayerType(self):

        self.__mutations[0] = self.__conv2d
        self.__mutations[1] = self.__linear

    def __generate_conv2d_mutations(self):

        self.__conv2d[0] = Conv2dMutations.AlterExitFilterMutation()
        self.__conv2d[1] = Conv2dMutations.AlterEntryFilterMutation()
        self.__conv2d[3] = Conv2dMutations.AlterDimensionKernel()
        
    def __generate_linear_mutations(self):
        
        self.__linear[0] = LinearMutations.AlterExitFilterMutation()
        self.__linear[1] = LinearMutations.AlterEntryFilterMutation()

    def __get_operation(self, oldFilter, newFilter):

        result = []
        for i in range(len(oldFilter.shape)):
            result.append(newFilter.shape[i] - oldFilter.shape[i])

        return tuple(result)    
    
    def __get_mutations_key_list(self, operation):

        index_list = []
        for i in range(len(operation)):
            if operation[i] != 0:
                index_list.append(i)

        return index_list
    
    def get_mutation_list(self, layerType, oldFilter, newFilter):

        operation = self.__get_operation(oldFilter, newFilter)
        
        mutations = self.__mutations.get(layerType)

        key_list = self.__get_mutations_key_list(operation)

        mutation_list = None

        if len(key_list) > 0:

            mutation_list = []
            for key in key_list:

                mutation_instance = mutations.get(key)

                if mutation_instance is not None:
                    mutation_instance.value = operation[key]
                    mutation_list.append(mutation_instance)

        return mutation_list