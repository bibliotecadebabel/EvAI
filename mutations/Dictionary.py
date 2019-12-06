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

        self.__mutationsConv2d[-1] = Conv2dMutations.RemoveFilterMutation()
        self.__mutationsConv2d[1] = Conv2dMutations.AddFilterMutation()

    def __generateMutationsLinear(self):

        self.__mutationsLinear[-1] = LinearMutations.RemoveFilterMutation()
        self.__mutationsLinear[1] = LinearMutations.AddFilterMutation()

    def getMutation(self, layerType, operation):

        mutationDict = self.__layerType[layerType]

        return mutationDict[operation]
