from mutations.MutationAbstract import Mutation
import torch as torch
import const.mutation_type as m_type

########## NORMAL MUTATIONS ##########

class AlterExitFilterMutation(Mutation):

    def __init__(self):
        super().__init__()

    _value = 0
    
    def value_getter(self):
        return self._value

    def value_setter(self, newvalue):
        self._value = newvalue

    value = property(value_getter, value_setter)
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):
        
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

            newNode.setFilter(oldFilter)
            newNode.setBias(oldBias)

            del resized
        
        elif self._value < 0:

            value = abs(self._value)

            oldFilter.resize_(old_shape[0]-value, old_shape[1], old_shape[2], old_shape[3])
            oldBias.resize_(old_shape[0]-value)

            newNode.setFilter(oldFilter)
            newNode.setBias(oldBias)

class AlterEntryFilterMutation(Mutation):

    def __init__(self):
        super().__init__()

    _value = 0
    
    def value_getter(self):
        return self._value

    def value_setter(self, newvalue):
        self._value = newvalue

    value = property(value_getter, value_setter)
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):

        old_shape = oldFilter.shape

        if self._value > 0:

            if cuda == True:
                resized = torch.zeros(old_shape[0], self._value, old_shape[2], old_shape[3]).cuda()
            else:
                resized = torch.zeros(old_shape[0], self._value, old_shape[2], old_shape[3])

            oldFilter = torch.cat((oldFilter, resized), dim=1)
            resized_shape = oldFilter.shape

            for i in range(old_shape[0]):
                for j in range(old_shape[1], resized_shape[1]):
                    oldFilter[i][j] = oldFilter[i][old_shape[1]-1].clone()

            newNode.setFilter(oldFilter)
            newNode.setBias(oldBias)
            
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

            del oldFilter

            newNode.setFilter(resized)
            newNode.setBias(oldBias)

class AlterDimensionKernel(Mutation):

    def __init__(self):
        super().__init__()

    _value = 0
    
    def value_getter(self):
        return self._value

    def value_setter(self, newvalue):
        self._value = newvalue

    value = property(value_getter, value_setter)
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):
        
        if self._value > 0:

            shape = oldFilter.shape

            if cuda == True:
                resized_1 = torch.zeros(shape[0], shape[1], shape[2], 1).cuda()
                resized_2 = torch.zeros(shape[0], shape[1], 1, shape[3]+1).cuda()
            else:
                resized_1 = torch.zeros(shape[0], shape[1], shape[2], 1)
                resized_2 = torch.zeros(shape[0], shape[1], 1, shape[3]+1)

            oldFilter = torch.cat((oldFilter, resized_1), dim=3)
            oldFilter = torch.cat((oldFilter, resized_2), dim=2)
            
            del resized_1
            del resized_2

            for out_channel in range(shape[0]):
                for in_channel in range(shape[1]):
                    oldFilter[out_channel][in_channel][shape[2]] = oldFilter[out_channel][in_channel][shape[2]-1].clone() 
                    for x in range(shape[2]+1):
                        oldFilter[out_channel][in_channel][x][shape[3]] = oldFilter[out_channel][in_channel][x][shape[3]-1].clone()

            newNode.setFilter(oldFilter)
            newNode.setBias(oldBias)
        
        elif self._value < 0:

            shape = oldFilter.shape

            if cuda == True:
                resized = torch.zeros(shape[0], shape[1], shape[2]-1, shape[3]-1).cuda()
            else:
                resized = torch.zeros(shape[0], shape[1], shape[2]-1, shape[3]-1)
            
            for out_channel in range(shape[0]):
                for in_channel in range(shape[1]):
                    for kernel_x in range(shape[2]-1):
                        for kernel_y in range(shape[3]-1):
                            resized[out_channel][in_channel][kernel_x][kernel_y] = oldFilter[out_channel][in_channel][kernel_x][kernel_y].clone()
        


            del oldFilter

            newNode.setFilter(resized)
            newNode.setBias(oldBias)

########## DENDRITE MUTATIONS ##########

class AlterExitFilterMutation_Dendrite(Mutation):

    def __init__(self):
        super().__init__()

    _value = 0
    
    def value_getter(self):
        return self._value

    def value_setter(self, newvalue):
        self._value = newvalue

    value = property(value_getter, value_setter)
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):
        
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

            newNode.setFilter(oldFilter)
            newNode.setBias(oldBias)

            del resized
        
        elif self._value < 0:

            value = abs(self._value)

            oldFilter.resize_(old_shape[0]-value, old_shape[1], old_shape[2], old_shape[3])
            oldBias.resize_(old_shape[0]-value)

            newNode.setFilter(oldFilter)
            newNode.setBias(oldBias)

class AlterEntryFilterMutation_Dendrite(Mutation):

    def __init__(self):
        super().__init__()

    _value = 0
    
    def value_getter(self):
        return self._value

    def value_setter(self, newvalue):
        self._value = newvalue

    value = property(value_getter, value_setter)
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):

        old_shape = oldFilter.shape

        if self._value > 0:

            if cuda == True:
                resized = torch.zeros(old_shape[0], self._value, old_shape[2], old_shape[3]).cuda()
            else:
                resized = torch.zeros(old_shape[0], self._value, old_shape[2], old_shape[3])

            oldFilter = torch.cat((oldFilter, resized), dim=1)
            resized_shape = oldFilter.shape

            '''
            for i in range(old_shape[0]):
                for j in range(old_shape[1], resized_shape[1]):
                    oldFilter[i][j] = oldFilter[i][old_shape[1]-1].clone()
            '''

            newNode.setFilter(oldFilter)
            newNode.setBias(oldBias)
            
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

            del oldFilter

            newNode.setFilter(resized)
            newNode.setBias(oldBias)

class AlterDimensionKernel_Dendrite(Mutation):

    def __init__(self):
        super().__init__()

    _value = 0
    
    def value_getter(self):
        return self._value

    def value_setter(self, newvalue):
        self._value = newvalue

    value = property(value_getter, value_setter)
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):

        newDimensions = abs(self._value)

        if self._value > 0:

            shape = oldFilter.shape

            if cuda == True:
                resized_1 = torch.zeros(shape[0], shape[1], shape[2], newDimensions).cuda()
                resized_2 = torch.zeros(shape[0], shape[1], newDimensions, shape[3]+newDimensions).cuda()
            else:
                resized_1 = torch.zeros(shape[0], shape[1], shape[2], newDimensions)
                resized_2 = torch.zeros(shape[0], shape[1], newDimensions, shape[3]+newDimensions)

            oldFilter = torch.cat((oldFilter, resized_1), dim=3)
            oldFilter = torch.cat((oldFilter, resized_2), dim=2)
            
            del resized_1
            del resized_2

            newNode.setFilter(oldFilter)
            newNode.setBias(oldBias)
        
        elif self._value < 0:

            shape = oldFilter.shape

            if cuda == True:
                resized = torch.zeros(shape[0], shape[1], shape[2]-newDimensions, shape[3]-newDimensions).cuda()
            else:
                resized = torch.zeros(shape[0], shape[1], shape[2]-newDimensions, shape[3]-newDimensions)
            
            for out_channel in range(shape[0]):
                for in_channel in range(shape[1]):
                    for kernel_x in range(shape[2]-newDimensions):
                        for kernel_y in range(shape[3]-newDimensions):
                            resized[out_channel][in_channel][kernel_x][kernel_y] = oldFilter[out_channel][in_channel][kernel_x][kernel_y].clone()
        


            del oldFilter

            newNode.setFilter(resized)
            newNode.setBias(oldBias)

class AdjustEntryFilters_Dendrite():

    # adjustLayer = Layer afectado que debe ser ajustado antes de pasar sus parametros al nuevo layer
    # indexList = Lista de index de los layers que envian filtros al layer afectado ordenados por jerarquia
    # targetIndex = Index del Layer objetivo de la mutacion, indica el indice de partida dentro del indexList
    # network = red neuronal donde se encuentra el layer afectado
    def __init__(self, adjustLayer, indexList, targetIndex, network):
        
        self.adjustLayer = adjustLayer
        self.indexList  = indexList
        self.targetIndex = targetIndex
        self.network = network

    # Only use when mutation is remove layer
    def removeFilters(self):

        print("oldLayer to adjust= ", self.adjustLayer.adn)
        print("index list=", self.indexList)
        print("target layer=", self.targetIndex)

        oldFilter = self.adjustLayer.getFilter()

        shape = oldFilter.shape

        value = self.getTargetRange()

        print("range to remove=", value)
        newEntries = shape[1] - (abs(value[0] - value[1]) + 1)

        if self.network.cudaFlag == True:
            adjustedOldFilter = torch.zeros(shape[0], newEntries, shape[2], shape[3]).cuda()
        else:
            adjustedOldFilter = torch.zeros(shape[0], newEntries, shape[2], shape[3])
        
        for exit_channel in range(shape[0]):
            index_accepted = 0
            for entries_channel in range(shape[1]):

                if entries_channel >= value[0] and entries_channel <= value[1]:
                    pass
                else:
                    adjustedOldFilter[exit_channel][index_accepted] = oldFilter[exit_channel][entries_channel].clone()
                    index_accepted += 1

        return adjustedOldFilter

    def adjustEntryFilters(self, oldFilter, newFilter,  mutation_type):
        
        value = oldFilter

        if mutation_type == m_type.DEFAULT_ADD_FILTERS:
            value =  self.__increaseEntryFilters(oldFilter=oldFilter, newFilter=newFilter)
        elif mutation_type == m_type.DEFAULT_REMOVE_FILTERS:
            value = self.__decreaseEntryFilters(oldFilter=oldFilter, newFilter=newFilter)
        
        return value

    def __decreaseEntryFilters(self, oldFilter, newFilter):
        
        shape = oldFilter.shape

        range_filter  = self.getTargetRange()
        value = abs(newFilter.shape[1] - oldFilter.shape[1])
        conserved_range = [range_filter[0], abs(range_filter[1] - value)]
        remove_gane = [conserved_range[1]+1, range_filter[1]]
        if self.network.cudaFlag == True:
            adjustedFilter = torch.zeros(shape[0], shape[1]-value, shape[2], shape[3]).cuda()
        else:
            adjustedFilter = torch.zeros(shape[0], shape[1]-value, shape[2], shape[3])
        
        print("layer range=", range_filter)
        print("conserved RANGE=", conserved_range)
        print("remove range=", remove_gane)
        print("new filter size=", adjustedFilter.shape)
        
        for exit_channel in range(shape[0]):
            index_accepted = 0
            for entries_channel in range(shape[1]):

                if entries_channel >= remove_gane[0] and entries_channel <= remove_gane[1]:
                    pass
                else:
                    adjustedFilter[exit_channel][index_accepted] = oldFilter[exit_channel][entries_channel].clone()
                    index_accepted += 1
        
        return adjustedFilter
    
    def __increaseEntryFilters(self, oldFilter, newFilter):
        startIndex = self.getTargetRange()[1]
        print("starting index to add=", startIndex)
        value = abs(newFilter.shape[1] - oldFilter.shape[1])

        range_add = [startIndex+1, startIndex+value]

        shape = oldFilter.shape

        if self.network.cudaFlag == True:
            adjustedFilter = torch.zeros(shape[0], shape[1]+value, shape[2], shape[3]).cuda()
        else:
            adjustedFilter = torch.zeros(shape[0], shape[1]+value, shape[2], shape[3])

        print("add range=", range_add)
        print("new filter size=", adjustedFilter.shape) 

        for exit_channel in range(shape[0]):
            index_accepted = 0
            for entries_channel in range(shape[1]+value):

                if entries_channel >= range_add[0] and entries_channel <= range_add[1]:
                    pass
                else:
                    adjustedFilter[exit_channel][entries_channel] = oldFilter[exit_channel][index_accepted].clone()
                    index_accepted += 1
        
        return adjustedFilter

    # Obtener el rango de indices del filtro de  layer afectado, dependiendo de la jerarquia del layer de la mutacion.
    def getTargetRange(self):

        starting = 0

        for index in self.indexList:
            
            if index == self.targetIndex:
                break
            else:
                indexNode = index + 1

                if indexNode > 0: #check if is not image
                    adn = self.network.nodes[indexNode].objects[0].adn
                    starting += adn[2]
                
                elif indexNode == 0:
                    starting += 3 #if indexNode equals zero is image, and image always has 3 output channels

        
        ending = self.network.nodes[self.targetIndex+1].objects[0].adn[2]

        value = [starting, starting + ending-1]

        return value