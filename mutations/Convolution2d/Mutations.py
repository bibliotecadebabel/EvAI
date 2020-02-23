from mutations.MutationAbstract import Mutation
import torch as torch

class AddExitFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):
        
        shape = oldFilter.shape

        if cuda == True:
            resized = torch.zeros(1, shape[1], shape[2], shape[3]).cuda()
        else:
            resized = torch.zeros(1, shape[1], shape[2], shape[3])

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
            resized = torch.zeros(shape[0], 1, shape[2], shape[3]).cuda()
        else:
            resized = torch.zeros(shape[0], 1, shape[2], shape[3])

        oldFilter = torch.cat((oldFilter, resized), dim=1)


        for i in range(shape[0]):
            oldFilter[i][shape[1]] = oldFilter[i][shape[1]-1].clone()

        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)
        
        del resized

class RemoveExitFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):

        shape = oldFilter.shape

        oldFilter.resize_(shape[0]-1, shape[1], shape[2], shape[3])
        oldBias.resize_(shape[0]-1)

        newNode.setFilter(oldFilter)
        newNode.setBias(oldBias)

class RemoveEntryFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):

        shape = oldFilter.shape

        if cuda == True:
            resized = torch.zeros(shape[0], shape[1]-1, shape[2], shape[3]).cuda()
        else:
            resized = torch.zeros(shape[0], shape[1]-1, shape[2], shape[3])

        for out_channel in range(shape[0]):
            for in_channel in range(shape[1]-1):
                resized[out_channel][in_channel] = oldFilter[out_channel][in_channel].clone()

        del oldFilter

        newNode.setFilter(resized)
        newNode.setBias(oldBias)

class AddDimensionKernel(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):
        
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
    
class RemoveDimensionKernel(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, oldFilter, oldBias, newNode, cuda):
        
        shape = oldFilter.shape

        if cuda == True:
            resized = torch.zeros(shape[0], shape[1], shape[2]-1, shape[3]-1).cuda()
        else:
            resized = torch.zeros(shape[0], shape[1], shape[2]-1, shape[3]-1)
        
        #oldFilter.resize_(shape[0], shape[1], shape[2], shape[3]-1, shape[4]-1)

        
        for out_channel in range(shape[0]):
            for in_channel in range(shape[1]):
                for kernel_x in range(shape[2]-1):
                    for kernel_y in range(shape[3]-1):
                        resized[out_channel][in_channel][kernel_x][kernel_y] = oldFilter[out_channel][in_channel][kernel_x][kernel_y].clone()
    


        del oldFilter

        newNode.setFilter(resized)
        newNode.setBias(oldBias)