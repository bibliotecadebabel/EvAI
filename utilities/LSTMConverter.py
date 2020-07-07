import Factory.TensorFactory as TensorFactory
import numpy as np

class LSTMConverter():

    def __init__(self, cuda, max_layers, limit_directions=3):
        self.cuda = cuda
        self.__createDictionary()
        self.mutations = len(self.mutation_to_index) - 1
        self.max_layers = max_layers
        self.limit_directions = limit_directions

    def __createDictionary(self):
        self.mutation_to_index = {}
        self.index_to_mutation = {}

        self.mutation_to_index[(1,1,1,1)] = -1 # space
        self.mutation_to_index[(1,0,0,0)] = 0
        self.mutation_to_index[(0,1,0,0)] = 1
        self.mutation_to_index[(4,0,0,0)] = 2
        self.mutation_to_index[(0,0,1)] = 3
        self.mutation_to_index[(0,0,-1)] = 4
        self.mutation_to_index[(0,0,2)] = 5
        self.mutation_to_index[(0,0,1,1)] = 6
        self.mutation_to_index[(0,0,-1,-1)] = 7
        
        self.index_to_mutation[-1] = (1,1,1,1)
        self.index_to_mutation[0] = (1,0,0,0)
        self.index_to_mutation[1] = (0,1,0,0)
        self.index_to_mutation[2] = (4,0,0,0)
        self.index_to_mutation[3] = (0,0,1)
        self.index_to_mutation[4] = (0,0,-1)
        self.index_to_mutation[5] = (0,0,2)
        self.index_to_mutation[6] = (0,0,1,1)
        self.index_to_mutation[7] = (0,0,-1,-1)

    def directionToTensor(self, direction):

        index_layer = direction[0]
        mutation = direction[1]
        index_mutation = self.mutation_to_index.get(mutation)

        if index_layer >= self.max_layers:
            raise Exception("index out of range: {:d} (max layers: {:d})".format(index_layer, self.max_layers))
        
        if index_mutation is None:
            raise Exception("mutation doesnt exist {}".format(mutation))

        value = TensorFactory.createTensorZeros(tupleShape=(self.max_layers, self.mutations), cuda=self.cuda)
        
        if index_mutation >= 0:
            value[index_layer, index_mutation] = 1

        return value
    
    def generateLSTMInput(self, observations):

        num_observations = len(observations)

        value = TensorFactory.createTensorZeros(tupleShape=(num_observations, self.limit_directions, self.max_layers, self.mutations), cuda=self.cuda)
      
        for observation in range(num_observations):

            directions = observations[observation].directions
            num_directions = len(directions)

            tensors_directions = []

            for i in range(num_directions):
                tensors_directions.append(self.directionToTensor(directions[i]))

            for i in range(num_directions, self.limit_directions):
                tensors_directions.append(self.directionToTensor(directions[num_directions-1]))
            
            for index in range(value.shape[1]):
                value[observation][index] = tensors_directions[index] 
        
        del tensors_directions
        
        return value

    def generateLSTMPredict(self, observation):

        value = TensorFactory.createTensorZeros(tupleShape=(1, self.limit_directions-1, self.max_layers, self.mutations), cuda=self.cuda)

        for i in range(self.limit_directions-1):
                value[0][i] = self.directionToTensor(observation.directions[i])

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





