from mutations.MutationAbstract import Mutation
import torch as torch

class AddExitFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):

        shape = oldFilter.shape
    
        if cuda == True:
            resized = torch.zeros(1, shape[1]).cuda()
        else:
            resized = torch.zeros(1, shape[1])

        
        oldFilter = torch.cat((oldFilter, resized), dim=0)
        oldBias.resize_(shape[0]+1)
        
        oldFilter[shape[0]] = oldFilter[shape[0]-1].clone()
        oldBias[shape[0]] = oldBias[shape[0]-1].clone()

        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias) 

        del resized

class AddEntryFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):
      
        shape = oldFilter.shape

        if cuda == True:
            resized = torch.zeros(shape[0], 1).cuda()
        else:
            resized = torch.zeros(shape[0], 1)

        oldFilter = torch.cat((oldFilter, resized), dim=1)

        for i in range(oldFilter.shape[0]):
            oldFilter[i][oldFilter.shape[1]-1] = oldFilter[i][oldFilter.shape[1]-2].clone()

        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)

        del resized


class RemoveExitFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):

        shape = oldFilter.shape

        oldFilter.resize_(shape[0]-1, shape[1])
        oldBias.resize_(shape[0]-1)

        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)

class RemoveEntryFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):
        
        shape = oldFilter.shape
    
        #oldFilter.resize_(shape[0], shape[1]-1)

        if cuda == True:
            resized = torch.zeros(shape[0], shape[1]-1).cuda()
        else:
            resized = torch.zeros(shape[0], shape[1]-1)
        
        for out_channel in range(shape[0]):
            for in_channel in range(shape[1]-1):
                resized[out_channel][in_channel] = oldFilter[out_channel][in_channel].clone()
        
        del oldFilter

        newNode.setFilter(resized)
        newNode.setBias(oldBias)