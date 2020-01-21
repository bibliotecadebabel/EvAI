from mutations.MutationAbstract import Mutation

class AddExitFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode):

        #print("Mutate AddExitFilterMutation Linear")
        
        shape = oldFilter.shape
        
        oldFilter.resize_(shape[0]+1, shape[1])
        oldFilter[shape[0]] = oldFilter[shape[0]-1].clone()

        oldBias.resize_(shape[0]+1)
        oldBias[shape[0]] = oldBias[shape[0]-1].clone()

        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)

class AddEntryFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode):

        #print("Mutate AddEntryFilterMutation Linear")
        
        shape = oldFilter.shape
        oldFilter.resize_(shape[0], shape[1]+1)

        for i in range(oldFilter.shape[0]):
            oldFilter[i][oldFilter.shape[1]-1] = oldFilter[i][oldFilter.shape[1]-2].clone()
        
        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)

class RemoveExitFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode):

        #print("Mutate RemoveExitFilterMutation Linear")
        
        shape = oldFilter.shape
        
        oldFilter.resize_(shape[0]-1, shape[1])
        oldBias.resize_(shape[0]-1)
        
        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)


class RemoveEntryFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode):
        
        #print("Mutate RemoveEntryFilterMutation Linear")

        shape = oldFilter.shape
        oldFilter.resize_(shape[0], shape[1]-1)
        
        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)