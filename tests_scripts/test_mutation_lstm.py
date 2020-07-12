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
    num_actions = 4
    directions_per_observations = [
                                    [(1,(1,0,0,0)), (2,(4,0,0,0)), (2,(1,0,0,0))],
                                    [(1,(1,0,0,0)), (2,(4,0,0,0)), (2,(4,0,0,0))],
                                    [(1,(1,0,0,0)), (2,(4,0,0,0)), (0,(0,1,0,0))],
                                    [(1,(1,0,0,0)), (2,(4,0,0,0)), (4,(0,0,1))]
                                ]
    #observations_weight = [getLoss(0.4525), getLoss(0.4520), getLoss(0.4145), getLoss(0.4220)]

    observations_weight = [0.0248*100, 0.0498*100, 0.0278*100, 0.0192*100]
    print("weight: ", observations_weight)

    lstmConverter = LSTMConverter.LSTMConverter(cuda=True, max_layers=max_layers_lstm, limit_directions=observation_size)
    observations = []

    for i in range(num_actions):
        observation = Observation.Observation(path=directions_per_observations[i], weight=observations_weight[i], time=0)
        print("observation: ", i)
        print("path: ", observation.path)
        print("weight: ", observation.weight)
        observations.append(observation)

    input_lstm = lstmConverter.generateLSTMInput(observations=observations)
    print(input_lstm.size())
    current_path = Observation.Observation(path=[(2,(4,0,0,0)), (2,(4,0,0,0))], weight=1, time=0)
    network = nw_lstm.NetworkLSTM(observation_size=observation_size, inChannels=max_layers_lstm, 
                                    outChannels=max_layers_lstm*lstmConverter.mutations, kernelSize=lstmConverter.mutations, cudaFlag=True)
    
    i = 0
    while i < 200 : 
        print("empezando entrenamiento")
        network.Training(data=input_lstm, dt=0.001, p=100, observations=observations)
        print("entrenamiento finalizado")

        predict = lstmConverter.generateLSTMPredict(observation=current_path)
        predicted = network.predict(predict)
        predicted_directions = lstmConverter.topKPredictedDirections(predicted_tensor=predicted, k=num_actions//2)
        print(predicted_directions)
        
        i += 1

