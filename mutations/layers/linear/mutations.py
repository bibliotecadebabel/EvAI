from mutations.layers.mutation import Mutation
import torch as torch

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

        if self._value > 0:

            shape = oldFilter.shape
        
            if cuda == True:
                resized = torch.zeros(1, shape[1]).cuda()
            else:
                resized = torch.zeros(1, shape[1])

            
            oldFilter = torch.cat((oldFilter, resized), dim=0)
            oldBias.resize_(shape[0]+1)
            
            oldFilter[shape[0]] = oldFilter[shape[0]-1].clone()
            oldBias[shape[0]] = oldBias[shape[0]-1].clone()

            newNode.set_filters(oldFilter)
            newNode.set_bias(oldBias) 

            del resized
        
        elif self._value < 0:

            shape = oldFilter.shape

            oldFilter.resize_(shape[0]-1, shape[1])
            oldBias.resize_(shape[0]-1)

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

        if self._value > 0:

            shape = oldFilter.shape

            if cuda == True:
                resized = torch.zeros(shape[0], self._value).cuda()
            else:
                resized = torch.zeros(shape[0], self._value)

            oldFilter = torch.cat((oldFilter, resized), dim=1)

            '''
            for i in range(oldFilter.shape[0]):
                oldFilter[i][oldFilter.shape[1]-1] = oldFilter[i][oldFilter.shape[1]-2].clone()
            '''
            newNode.set_filters(oldFilter)
            newNode.set_bias(oldBias)

            del resized
        
        elif self._value < 0:

            shape = oldFilter.shape
            newDimension = abs(self._value)

            if cuda == True:
                resized = torch.zeros(shape[0], shape[1]-newDimension).cuda()
            else:
                resized = torch.zeros(shape[0], shape[1]-newDimension)
            
            for out_channel in range(shape[0]):
                for in_channel in range(shape[1]-newDimension):
                    resized[out_channel][in_channel] = oldFilter[out_channel][in_channel].clone()
            
            del oldFilter

            newNode.set_filters(resized)
            newNode.set_bias(oldBias)
        
