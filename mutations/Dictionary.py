import mutations.Convolution2d.Mutations as Conv2dMutations
import mutations.Linear.Mutations as LinearMutations

class MutationsDictionary():

    def __init__(self):

        self.__layerType = {}
        self.__layerType_dendrite = {}
        
        self.__mutationsConv2d = {}
        self.__mutationsConv2d_dendrite = {}

        self.__mutationsLinear = {}
        self.__mutationsCrossEntropy = {}
        
        self.__generateMutationsConv2d()
        self.__generateMutationsConv2d_dendrite()
        
        self.__generateMutationsLinear()

        self.__generateLayerType()
        self.__generateLayerType_dendrite()

    def __generateLayerType(self):

        self.__layerType[0] = self.__mutationsConv2d
        self.__layerType[1] = self.__mutationsLinear
        self.__layerType[2] = self.__mutationsCrossEntropy

    def __generateLayerType_dendrite(self):

        self.__layerType_dendrite[0] = self.__mutationsConv2d_dendrite
        self.__layerType_dendrite[1] = self.__mutationsLinear
        self.__layerType_dendrite[2] = self.__mutationsCrossEntropy


    def __generateMutationsConv2d(self):

        self.__mutationsConv2d[1] = Conv2dMutations.AlterEntryFilterMutation()
        self.__mutationsConv2d[2] = Conv2dMutations.AlterExitFilterMutation()
        self.__mutationsConv2d[4] = Conv2dMutations.AlterDimensionKernel()

    def __generateMutationsConv2d_dendrite(self):

        self.__mutationsConv2d_dendrite[0] = Conv2dMutations.AlterExitFilterMutation_Dendrite()
        self.__mutationsConv2d_dendrite[1] = Conv2dMutations.AlterEntryFilterMutation_Dendrite()
        self.__mutationsConv2d_dendrite[3] = Conv2dMutations.AlterDimensionKernel_Dendrite()
        

    def __generateMutationsLinear(self):
        
        self.__mutationsLinear[0] = LinearMutations.AlterExitFilterMutation()
        self.__mutationsLinear[1] = LinearMutations.AlterEntryFilterMutation()

    def __getOperation(self, oldAdn, newAdn):

        result = list(oldAdn).copy()

        for i in range(len(oldAdn)):
            result[i] = newAdn[i] - oldAdn[i]

        return tuple(result) 

    def __getOperationByFilter(self, oldFilter, newFilter):

        result = []
        for i in range(len(oldFilter.shape)):
            result.append(newFilter.shape[i] - oldFilter.shape[i])

        return tuple(result)    
    
    def __getMutationKey(self, operation):
        
        index = 0
        for i in range(len(operation)):
            if operation[i] != 0:
                index = i

        return index
    
    def __getMutationKeyList(self, operation):

        index_list = []
        for i in range(len(operation)):
            if operation[i] != 0:
                index_list.append(i)

        return index_list

    def getMutation(self, oldAdn, newAdn):

        layerType = oldAdn[0]
        operation = self.__getOperation(oldAdn, newAdn)

        mutationDict = self.__layerType.get(layerType)

        key = self.__getMutationKey(operation)

        mutationValue = mutationDict.get(key)

        if mutationValue is not None:
            mutationValue.value = operation[key]

        return mutationValue
    
    def getMutationList(self, layerType, oldFilter, newFilter):

        operation = self.__getOperationByFilter(oldFilter, newFilter)
        
        mutationDict_dendrite = self.__layerType_dendrite.get(layerType)

        key_list = self.__getMutationKeyList(operation)

        mutation_list = None

        if len(key_list) > 0:

            mutation_list = []
            for key in key_list:

                mutation_instance = mutationDict_dendrite.get(key)

                if mutation_instance is not None:
                    mutation_instance.value = operation[key]
                    mutation_list.append(mutation_instance)

        return mutation_list