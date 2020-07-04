import Factory.TensorFactory as TensorFactory
import numpy as np

class LSTMConverter():

    def __init__(self, cuda, max_layers):
        self.cuda = cuda
        self.__createDictionary()
        self.mutations = len(self.mutation_to_index)
        self.max_layers = max_layers

    def __createDictionary(self):
        self.mutation_to_index = {}
        self.index_to_mutation = {}

        self.mutation_to_index[(1,0,0,0)] = 0
        self.mutation_to_index[(0,1,0,0)] = 1
        self.mutation_to_index[(4,0,0,0)] = 2
        self.mutation_to_index[(0,0,1)] = 3
        self.mutation_to_index[(0,0,-1)] = 4
        self.mutation_to_index[(0,0,2)] = 5
        self.mutation_to_index[(0,0,1,1)] = 6
        self.mutation_to_index[(0,0,-1,-1)] = 7

        self.index_to_mutation[0] = (1,0,0,0)
        self.index_to_mutation[1] = (0,1,0,0)
        self.index_to_mutation[2] = (4,0,0,0)
        self.index_to_mutation[3] = (0,0,1)
        self.index_to_mutation[4] = (0,0,-1)
        self.index_to_mutation[5] = (0,0,2)
        self.index_to_mutation[6] = (0,0,1,1)
        self.index_to_mutation[7] = (0,0,-1,-1)


    def __mutationToArray(self, mutation):

        value = None
        index = self.mutation_to_index.get(mutation)

        if index is not None:
            value = TensorFactory.createTensorZeros(tupleShape=(self.mutations), cuda=self.cuda)
            value[index] = 1
        else:
            raise Exception("mutation doesnt exist {}".format(mutation))
        
        return value
    
    def __indexLayerToArray(self, index):

        if index >= self.max_layers:
            raise Exception("index out of range: {:d} (max layers: {:d})".format(index, self.max_layers))
        
        value = TensorFactory.createTensorZeros(tupleShape=(self.max_layers), cuda=self.cuda)
        value[index] = 1

        return value

    def directionToTensor(self, direction):

        index_layer = direction[0]
        mutation = direction[1]
        index_mutation = self.mutation_to_index.get(mutation)

        if index_layer >= self.max_layers:
            raise Exception("index out of range: {:d} (max layers: {:d})".format(index_layer, self.max_layers))
        
        if index_mutation is None:
            raise Exception("mutation doesnt exist {}".format(mutation))

        value = TensorFactory.createTensorZeros(tupleShape=(self.max_layers, self.mutations), cuda=self.cuda)
        
        value[index_layer, index_mutation] = 1

        return value
    
    def tensorToDirection(self, tensor):

        index_values = []
        for index_layer in range(tensor.shape[0]):

            for index_mutation in range(tensor.shape[1]):

                if tensor[index_layer][index_mutation] == 1:

                    index_values.append(index_layer)
                    index_values.append(index_mutation)
                    break
        
        mutation = self.index_to_mutation.get(index_values[1])

        if mutation is None or index_values[0] >= self.max_layers:
            raise Exception("Error converting tensor to direction {}".format(tensor))

        return (index_values[0],mutation)






