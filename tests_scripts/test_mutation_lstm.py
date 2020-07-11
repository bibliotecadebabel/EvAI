import utilities.LSTMConverter as LSTMConverter
import Geometric.Observations.Observation as Observation
import LSTM.NetworkLSTM as nw_lstm
import math
import torch

def getLoss(loss):

    value = 1/loss
    print(value)
    return round(value, 4)

def test():

    observation_size = 3
    max_layers_lstm = 20
    num_actions = 2
    directions_per_observations = [
                                    [(1,(1,0,0,0)), (2,(4,0,0,0)), (1,(1,0,0,0))],
                                    [(2,(4,0,0,0)), (1,(1,0,0,0)), (2,(4,0,0,0))],
                                    #[(2,(0,1,0,0)),(7,(0,0,1,1))],
                                    #[(19,(4,0,0,0)),(5,(4,0,0,0))],
                                    #[(9,(0,0,1)),(11,(0,0,2))]
                                ]
    #observations_weight = [getLoss(0.4525), getLoss(0.4520), getLoss(0.4145), getLoss(0.4220)]

    observations_weight = [1, 1]
    print("weight: ", observations_weight)

    lstmConverter = LSTMConverter.LSTMConverter(cuda=True, max_layers=max_layers_lstm, limit_directions=observation_size)
    observations = []

    for i in range(num_actions):
        observation = Observation.Observation()
        print("observation: ", i)
        for direction in directions_per_observations[i]:
            print(direction)
            observation.directions.append(direction)
        
        observation.weight = observations_weight[i]
        observations.append(observation)

    input_lstm = lstmConverter.generateLSTMInput(observations=observations)
    shape = input_lstm.shape
    print(input_lstm.size())

    network = nw_lstm.NetworkLSTM(observation_size=observation_size, inChannels=max_layers_lstm, 
                                    outChannels=max_layers_lstm*lstmConverter.mutations, kernelSize=lstmConverter.mutations, cudaFlag=True)
    
    print("empezando entrenamiento")
    network.Training(data=input_lstm, dt=0.001, p=2000, observations=observations)
    print("entrenamiento finalizado")


    stop = False
    
    for observation in observations:
        predict = lstmConverter.generateLSTMPredict(observation=observations)
        #print("predict: ", predict.size())
        predicted = network.predict(predict)
        #print("size: ", predicted[0].size())
        lstmConverter.topKPredictedDirections(predicted_tensor=predicted[0], k=2)
        print(predicted[1])
        #direction_predicted = lstmConverter.predictedToDirection(predicted_values=predicted)
        #print("## PREDICTED: ", direction_predicted)
    

