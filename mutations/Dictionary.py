import mutations.Convolution2d.Mutations as Conv2dMutations
import mutations.Linear.Mutations as LinearMutations

class MutationsDictionary():

    def __init__(self):

        self.__layerType = {}
        self.__mutationsConv2d = {}
        self.__mutationsLinear = {}
        self.__mutationsCrossEntropy = {}

        self.__generateLayerType()
        self.__generateMutationsConv2d()
        self.__generateMutationsLinear()

    def __generateLayerType(self):

        self.__layerType[0] = self.__mutationsConv2d
        self.__layerType[1] = self.__mutationsLinear
        self.__layerType[2] = self.__mutationsCrossEntropy

    def __generateMutationsConv2d(self):

        self.__mutationsConv2d[1] = Conv2dMutations.AlterEntryFilterMutation()
        self.__mutationsConv2d[2] = Conv2dMutations.AlterExitFilterMutation()
        self.__mutationsConv2d[4] = Conv2dMutations.AlterDimensionKernel()
        #self.__mutationsConv2d[(0, 0, -1, 0, 0)] = Conv2dMutations.RemoveExitFilterMutation()
        #self.__mutationsConv2d[(0, -1, 0, 0, 0)] = Conv2dMutations.RemoveEntryFilterMutation()
        #self.__mutationsConv2d[(0, 0, 0, -1, -1)] = Conv2dMutations.RemoveDimensionKernel()
        

    def __generateMutationsLinear(self):
        
        self.__mutationsLinear[1] = LinearMutations.AlterEntryFilterMutation()
        self.__mutationsLinear[2] = LinearMutations.AlterExitFilterMutation()

    def __getOperation(self, oldAdn, newAdn):

        result = list(oldAdn).copy()

        for i in range(len(oldAdn)):
            result[i] = newAdn[i] - oldAdn[i]

        return tuple(result)    
    
    def __getMutationKey(self, operation):
        
        index = 0
        for i in range(len(operation)):
            if operation[i] != 0:
                index = i

        return index
            
    def getMutation(self, oldAdn, newAdn):

        layerType = oldAdn[0]
        operation = self.__getOperation(oldAdn, newAdn)

        mutationDict = self.__layerType.get(layerType)

        key = self.__getMutationKey(operation)

        mutationValue = mutationDict.get(key)

        if mutationValue is not None:
            mutationValue.value = operation[key]

        return mutationValue
