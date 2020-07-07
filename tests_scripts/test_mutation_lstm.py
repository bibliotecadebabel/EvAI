import utilities.LSTMConverter as LSTMConverter
import Geometric.Observations.Observation as Observation
import LSTM.NetworkLSTM as nw_lstm
def test():


    num_actions = 4
    directions_per_observations = [
                                    [(1,(1,0,0,0)), (3,(4,0,0,0))],
                                    [(2,(0,1,0,0)),(7,(0,0,1,1))],
                                    [(19,(4,0,0,0)),(5,(4,0,0,0))],
                                    [(9,(0,0,1)),(11,(0,0,2))]
                                ]
    lstmConverter = LSTMConverter.LSTMConverter(cuda=True, max_layers=20, limit_directions=3)
    observations = []

    for i in range(num_actions):
        observation = Observation.Observation()
        print("observation: ", i)
        for direction in directions_per_observations[i]:
            print(direction)
            observation.directions.append(direction)
        
        observations.append(observation)

    input_lstm = lstmConverter.generateLSTMInput(observations=observations)
    shape = input_lstm.shape
    print(input_lstm.size())

    network = nw_lstm.NetworkLSTM(max_letters=shape[1], inChannels=shape[2], 
                                    outChannels=shape[2]*8, kernelSize=shape[3], cudaFlag=True)
    
    print("empezando entrenamiento")
    network.Training(data=input_lstm, dt=0.001, p=100)
    print("entrenamiento finalizado")


    stop = False
    
    for observation in observations:
        predict = lstmConverter.generateLSTMPredict(observation=observation)
        print("predict: ", predict.size())
        network.predict(predict)
    

