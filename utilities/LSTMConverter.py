import Factory.TensorFactory as TensorFactory
import numpy as np
import const.general_values as const_values
import torch

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

        self.mutation_to_index[const_values.EMPTY_MUTATION] = const_values.EMPTY_INDEX_LAYER
        self.mutation_to_index[(1,0,0,0)] = 0
        self.mutation_to_index[(0,1,0,0)] = 1
        self.mutation_to_index[(4,0,0,0)] = 2
        self.mutation_to_index[(0,0,1)] = 3
        self.mutation_to_index[(0,0,-1)] = 4
        self.mutation_to_index[(0,0,2)] = 5
        self.mutation_to_index[(0,0,1,1)] = 6
        self.mutation_to_index[(0,0,-1,-1)] = 7
        
        self.index_to_mutation[const_values.EMPTY_INDEX_LAYER] = const_values.EMPTY_MUTATION
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

            directions = observations[observation].path
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
        
        directions_number = len(observation.path)

        value = TensorFactory.createTensorZeros(tupleShape=(1, directions_number, self.max_layers, self.mutations), cuda=self.cuda)

        for i in range(directions_number):
            
            print("direction: ", observation.path[i])
            value[0][i] = self.directionToTensor(observation.path[i])

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
    
    def predictedToDirection(self, predicted_layer_index, predicted_mutation_index):

        mutation = self.index_to_mutation.get(predicted_mutation_index)

        return (predicted_layer_index, mutation)

    def topKPredictedDirections(self, predicted_tensor, k=2):
        
        shape = predicted_tensor.shape
        tensor = predicted_tensor.view(shape[0], 1, shape[1]*shape[2])
        predicted_directions = []

        topk_tensors = torch.topk(tensor, k=k)
        predicted_indexs_tensor = topk_tensors[1].view(-1)
        
        for i in range(predicted_indexs_tensor.shape[0]):

            index = predicted_indexs_tensor[i].item()

            top_i_layer_index = index // shape[2]
            top_i_mutation_index = index - (shape[2]*top_i_layer_index)
            predicted_direction = self.predictedToDirection(top_i_layer_index, top_i_mutation_index)
            predicted_directions.append(predicted_direction)

        return predicted_directions

