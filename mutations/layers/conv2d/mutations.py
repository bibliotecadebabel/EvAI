from mutations.layers.mutation import Mutation
import torch as torch
import const.mutation_type as m_type

class OutputChannelMutation(Mutation):

    def __init__(self):
        super().__init__()

    _value = 0
    
    def value_getter(self):
        return self._value

    def value_setter(self, newvalue):
        self._value = newvalue

    value = property(value_getter, value_setter)
    
    def execute(self, oldFilter, oldBias, newNode, cuda):
        
        old_shape = oldFilter.shape

        if self._value > 0:

            if cuda == True:
                resized = torch.zeros(self._value, old_shape[1], old_shape[2], old_shape[3]).cuda()
            else:
                resized = torch.zeros(self._value, old_shape[1], old_shape[2], old_shape[3])

            oldFilter = torch.cat((oldFilter, resized), dim=0)

            oldBias.resize_(old_shape[0]+self._value)

            resized_shape = oldFilter.shape

            for i in range(old_shape[0], resized_shape[0]):
                oldFilter[i] = oldFilter[old_shape[0]-1].clone()
                oldBias[i] = oldBias[old_shape[0]-1].clone()

            newNode.set_filters(oldFilter)
            newNode.set_bias(oldBias)

            del resized
        
        elif self._value < 0:

            value = abs(self._value)

            oldFilter.resize_(old_shape[0]-value, old_shape[1], old_shape[2], old_shape[3])
            oldBias.resize_(old_shape[0]-value)

            newNode.set_filters(oldFilter)
            newNode.set_bias(oldBias)

class InputChannelMutation(Mutation):

    def __init__(self):
        super().__init__()

    _value = 0
    
    def value_getter(self):
        return self._value

    def value_setter(self, newvalue):
        self._value = newvalue

    value = property(value_getter, value_setter)
    
    def execute(self, oldFilter, oldBias, newNode, cuda):

        old_shape = oldFilter.shape
        new_filter_shape = newNode.get_filters().shape

        if self._value > 0:

            if cuda == True:
                resized = torch.zeros(old_shape[0], self._value, old_shape[2], old_shape[3]).cuda()
            else:
                resized = torch.zeros(old_shape[0], self._value, old_shape[2], old_shape[3])

            oldFilter = torch.cat((oldFilter, resized), dim=1)
            resized_shape = oldFilter.shape

            '''            
            if str(new_filter_shape) == str(resized_shape):
                
                old_normalize = old_shape[1] * old_shape[2] * old_shape[3]
                new_normalize = resized_shape[1] * resized_shape[2] * resized_shape[3]
                oldFilter = (oldFilter * new_normalize) / old_normalize
                oldBias = (oldBias * new_normalize) / old_normalize
            '''

            newNode.set_filters(oldFilter)
            newNode.set_bias(oldBias)

            del resized

        elif self._value < 0:
            
            value = abs(self._value)

            if cuda == True:
                resized = torch.zeros(old_shape[0], old_shape[1]-value, old_shape[2], old_shape[3]).cuda()
            else:
                resized = torch.zeros(old_shape[0], old_shape[1]-value, old_shape[2], old_shape[3])

            for out_channel in range(old_shape[0]):
                for in_channel in range(old_shape[1]-value):
                    resized[out_channel][in_channel] = oldFilter[out_channel][in_channel].clone()

            '''
            if str(new_filter_shape) == str(resized.shape):
                
                old_normalize = old_shape[1] * old_shape[2] * old_shape[3]
                new_normalize = resized.shape[1] * resized.shape[2] * resized.shape[3]
                resized = (resized * new_normalize) / old_normalize
                oldBias = (oldBias * new_normalize) / old_normalize
            '''
            del oldFilter

            newNode.set_filters(resized)
            newNode.set_bias(oldBias)

class KernelSizeMutation(Mutation):

    def __init__(self):
        super().__init__()

    _value = 0
    
    def value_getter(self):
        return self._value

    def value_setter(self, newvalue):
        self._value = newvalue

    value = property(value_getter, value_setter)
    
    def execute(self, oldFilter, oldBias, newNode, cuda):

        newDimensions = abs(self._value)

        if self._value > 0:

            shape = oldFilter.shape
            old_normalize = 1 * shape[2] * shape[3]

            if cuda == True:
                resized_1 = torch.zeros(shape[0], shape[1], shape[2], newDimensions).cuda()
                resized_2 = torch.zeros(shape[0], shape[1], newDimensions, shape[3]+newDimensions).cuda()
            else:
                resized_1 = torch.zeros(shape[0], shape[1], shape[2], newDimensions)
                resized_2 = torch.zeros(shape[0], shape[1], newDimensions, shape[3]+newDimensions)

            oldFilter = torch.cat((oldFilter, resized_1), dim=3)
            oldFilter = torch.cat((oldFilter, resized_2), dim=2)
            
            #new_normalize = 1 * oldFilter.shape[2] * oldFilter.shape[3]

            del resized_1
            del resized_2

            
            #oldFilter = (oldFilter * new_normalize) / old_normalize

            #oldBias = (oldBias * new_normalize) / old_normalize
            

            newNode.set_filters(oldFilter)
            newNode.set_bias(oldBias)
        
        elif self._value < 0:

            shape = oldFilter.shape
            #old_normalize = 1 * shape[2] * shape[3]

            new_x = shape[2] - newDimensions
            new_y = shape[3] - newDimensions
            resized = oldFilter[:, :, :new_x, :new_y]

            del oldFilter

            #new_normalize = 1 * resized.shape[2] * resized.shape[3]
            
            #resized = (resized * new_normalize) / old_normalize

            #oldBias = (oldBias * new_normalize) / old_normalize
            
            newNode.set_filters(resized)
            newNode.set_bias(oldBias)

class AdjustInputChannels():

    # adjustLayer = Layer afectado que debe ser ajustado antes de pasar sus parametros al nuevo layer
    # indexList = Lista de index de los layers que envian filtros al layer afectado ordenados por jerarquia
    # targetIndex = Index del Layer objetivo de la mutacion, indica el indice de partida dentro del indexList
    # network = red neuronal donde se encuentra el layer afectado
    def __init__(self, adjustLayer, indexList, targetIndex, network, newFilter):
        
        self.adjustLayer = adjustLayer
        self.indexList  = indexList
        self.targetIndex = targetIndex
        self.network = network
        self.newFilter = newFilter

    # Only use when mutation is remove layer or remove dendrite
    def removeFilters(self):

        oldFilter = self.adjustLayer.get_filters()
        oldBias = self.adjustLayer.get_bias()

        shape = oldFilter.shape

        value = self.getTargetRange()

        #print("range to remove=", value)
        newEntries = shape[1] - (abs(value[0] - value[1]) + 1)
        #print("new entries: ", newEntries)

        adjustedOldFilter = self.__generateEmptyFilters(oldShape=shape, inputChannels=newEntries, cuda=self.network.cuda_flag)
        
        for output_channel in range(shape[0]):
            index_accepted = 0
            for input_channel in range(shape[1]):

                if input_channel >= value[0] and input_channel <= value[1]:
                    pass
                else:
                    adjustedOldFilter[output_channel][index_accepted] = oldFilter[output_channel][input_channel].clone()
                    index_accepted += 1

        value = self.__normalize(oldFilter=adjustedOldFilter, oldBias=oldBias, originalShape=shape)
        #print("value shape: ", value[0].shape)
        return value

    def adjustEntryFilters(self, mutation_type):
        
        value = [self.adjustLayer.get_filters(), self.adjustLayer.get_bias()]

        if mutation_type == m_type.DEFAULT_ADD_FILTERS:
            value =  self.__increaseEntryFilters()
        elif mutation_type == m_type.DEFAULT_REMOVE_FILTERS:
            value = self.__decreaseEntryFilters()
        elif mutation_type == m_type.DEFAULT_REMOVE_DENDRITE:
            value = self.removeFilters()
        
        return value

    def __decreaseEntryFilters(self):
        
        oldFilter = self.adjustLayer.get_filters()
        oldBias = self.adjustLayer.get_bias()
        newFilter = self.newFilter

        shape = oldFilter.shape

        range_filter  = self.getTargetRange()
        value = abs(newFilter.shape[1] - oldFilter.shape[1])
        conserved_range = [range_filter[0], abs(range_filter[1] - value)]
        remove_gane = [conserved_range[1]+1, range_filter[1]]

        adjustedFilter = self.__generateEmptyFilters(oldShape=shape, inputChannels=shape[1]-value, cuda=self.network.cuda_flag)
        
        '''
        print("layer range=", range_filter)
        print("conserved RANGE=", conserved_range)
        print("range to remove=", remove_gane)
        '''
        for output_channel in range(shape[0]):
            index_accepted = 0
            for input_channel in range(shape[1]):

                if input_channel >= remove_gane[0] and input_channel <= remove_gane[1]:
                    pass
                else:
                    adjustedFilter[output_channel][index_accepted] = oldFilter[output_channel][input_channel].clone()
                    index_accepted += 1
        
        value = self.__normalize(oldFilter=adjustedFilter, oldBias=oldBias, originalShape=shape)
        return value
    
    def __increaseEntryFilters(self):

        startIndex = self.getTargetRange()[1]

        oldFilter = self.adjustLayer.get_filters()
        oldBias = self.adjustLayer.get_bias()
        newFilter = self.newFilter

        value = abs(newFilter.shape[1] - oldFilter.shape[1])

        range_add = [startIndex+1, startIndex+value]

        #print("range to add: ", range_add)
        #print("value: ", value)
        shape = oldFilter.shape

        adjustedFilter = self.__generateEmptyFilters(oldShape=shape, inputChannels=shape[1]+value, cuda=self.network.cuda_flag)
        
        '''
        print("adjustfilter: ", adjustedFilter.size())
        print("oldfilter: ", oldFilter.size())
        print("add range=", range_add)
        print("new filter size=", adjustedFilter.shape) 
        '''

        for output_channel in range(shape[0]):
            index_accepted = 0
            for input_channel in range(shape[1]+value):

                if input_channel >= range_add[0] and input_channel <= range_add[1]:
                    pass
                else:
                    new_value = oldFilter[output_channel][index_accepted].clone()
                    adjustedFilter[output_channel][input_channel] = new_value
                    index_accepted += 1
        
        value = self.__normalize(oldFilter=adjustedFilter, oldBias=oldBias, originalShape=shape)
        return value

    # Obtener el rango de indices del filtro de  layer afectado, dependiendo de la jerarquia del layer de la mutacion.
    def getTargetRange(self):

        starting = 0

        for index in self.indexList:
            
            if index == self.targetIndex:
                break
            else:
                indexNode = index + 1

                if indexNode > 0: #check if is not image
                    dna = self.network.nodes[indexNode].objects[0].dna
                    starting += dna[2]
                
                elif indexNode == 0:
                    starting += 3 #if indexNode equals zero is image, and image always has 3 output channels

        
        if self.targetIndex+1 == 0:
            ending = 3
        else:
            ending = self.network.nodes[self.targetIndex+1].objects[0].dna[2]

        value = [starting, starting + ending-1]

        return value
    
    def __normalize(self, oldFilter, oldBias, originalShape):
        
        return [oldFilter, oldBias]
    
    def __generateEmptyFilters(self, oldShape, inputChannels, cuda):
        
        emptyFilters = None

        if len(oldShape) > 2: # Conv2d layer
            
            if cuda == True:
                emptyFilters = torch.zeros(oldShape[0], inputChannels, oldShape[2], oldShape[3]).cuda()
            else:
                emptyFilters = torch.zeros(oldShape[0], inputChannels, oldShape[2], oldShape[3])

        else: # Linear layer

            if cuda == True:
                emptyFilters = torch.zeros(oldShape[0], inputChannels).cuda()
            else:
                emptyFilters = torch.zeros(oldShape[0], inputChannels)

        return emptyFilters