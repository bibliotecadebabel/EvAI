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

        self.__mutationsConv2d[(0, 0, -1, 0, 0)] = Conv2dMutations.RemoveFilterMutation()
        self.__mutationsConv2d[(0, 0, 1, 0, 0)] = Conv2dMutations.AddFilterMutation()

    def __generateMutationsLinear(self):

        self.__mutationsLinear[(0, -1, 0)] = LinearMutations.RemoveFilterMutation()
        self.__mutationsLinear[(0, 1, 0)] = LinearMutations.AddFilterMutation()

    def __getOperation(self, oldAdn, newAdn):

        result = list(oldAdn).copy()

        for i in range(len(oldAdn)):
            result[i] = newAdn[i] - oldAdn[i]

        return tuple(result)    
            
    def getMutation(self, oldAdn, newAdn):

        layerType = oldAdn[0]
        operation = self.__getOperation(oldAdn, newAdn)

        print(operation)

        mutationDict = self.__layerType[layerType]

        return mutationDict[operation]
