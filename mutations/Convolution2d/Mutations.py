from mutations.MutationAbstract import Mutation

class AddExitFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode):
        
        #print("Mutate AddExitFilterMutation Conv2d")

        shape = oldFilter.shape

        oldFilter.resize_(shape[0]+1, shape[1], shape[2], shape[3])
        oldBias.resize_(shape[0]+1)

        oldFilter[shape[0]] = oldFilter[shape[0]-1].clone()
        oldBias[shape[0]] = oldBias[shape[0]-1].clone()

        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)

class AddEntryFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode):
        
        #print("Mutate AddEntryFilterMutation Conv2d")

        shape = oldFilter.shape

        oldFilter.resize_(shape[0], shape[1]+1, shape[2], shape[3])

        oldFilter[shape[1]] = oldFilter[shape[1]-1].clone()

        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)
        
class RemoveExitFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode):

        #print("Mutate RemoveExitFilterMutation Conv2d")

        shape = oldFilter.shape

        oldFilter.resize_(shape[0]-1, shape[1], shape[2], shape[3])
        oldBias.resize_(shape[0]-1)

        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)

class RemoveEntryFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode):
        
        #print("Mutate RemoveEntryFilterMutation Conv2d")

        shape = oldFilter.shape

        oldFilter.resize_(shape[0], shape[1]-1, shape[2], shape[3])

        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)