import Factory.TensorFactory as TensorFactory
import numpy as np
import const.general_values as const_values
import torch

class LSTMConverter():

    def __init__(self, cuda, max_layers, mutation_list, limit_directions=3):
        self.cuda = cuda
        self.__mutation_list = mutation_list
        self.__create_dictionary()
        self.mutations = len(self.mutation_to_index) - 1
        self.max_layers = max_layers
        self.limit_directions = limit_directions

    def __create_dictionary(self):

        index_mutation = 0
        self.mutation_to_index = {}
        self.index_to_mutation = {}

        self.mutation_to_index[const_values.EMPTY_MUTATION] = const_values.EMPTY_INDEX_LAYER
        self.index_to_mutation[const_values.EMPTY_INDEX_LAYER] = const_values.EMPTY_MUTATION

        for mutation in self.__mutation_list:

            if mutation not in self.mutation_to_index:

                self.mutation_to_index[mutation] = index_mutation
                self.index_to_mutation[index_mutation] = mutation
                index_mutation += 1

    def __direction_to_tensor(self, direction):

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
    
    def generate_LSTM_input(self, observations):

        num_observations = len(observations)

        value = TensorFactory.createTensorZeros(tupleShape=(num_observations, self.limit_directions, self.max_layers, self.mutations), cuda=self.cuda)
      
        for observation in range(num_observations):

            directions = observations[observation].path
            num_directions = len(directions)

            tensors_directions = []

            for i in range(num_directions):
                tensors_directions.append(self.__direction_to_tensor(directions[i]))

            for i in range(num_directions, self.limit_directions):
                tensors_directions.append(self.__direction_to_tensor(directions[num_directions-1]))
            
            for index in range(value.shape[1]):
                value[observation][index] = tensors_directions[index] 
        
        del tensors_directions
        
        return value

    def generate_LSTM_predict(self, observation):
        
        directions_number = len(observation.path)

        value = TensorFactory.createTensorZeros(tupleShape=(1, directions_number, self.max_layers, self.mutations), cuda=self.cuda)

        for i in range(directions_number):
            
            value[0][i] = self.__direction_to_tensor(observation.path[i])

        return value           
    
    def __predicted_to_direction(self, predicted_layer_index, predicted_mutation_index):

        mutation = self.index_to_mutation.get(predicted_mutation_index)

        return (predicted_layer_index, mutation)

    def topK_predicted_directions(self, predicted_tensor, k=2):
        
        shape = predicted_tensor.shape
        tensor = predicted_tensor.view(shape[0], 1, shape[1]*shape[2])
        predicted_directions = []

        topk_tensors = torch.topk(tensor, k=k)
        predicted_indexs_tensor = topk_tensors[1].view(-1)
        
        for i in range(predicted_indexs_tensor.shape[0]):

            index = predicted_indexs_tensor[i].item()

            top_i_layer_index = index // shape[2]
            top_i_mutation_index = index - (shape[2]*top_i_layer_index)
            predicted_direction = self.__predicted_to_direction(top_i_layer_index, top_i_mutation_index)
            predicted_directions.append(predicted_direction)

        return predicted_directions

