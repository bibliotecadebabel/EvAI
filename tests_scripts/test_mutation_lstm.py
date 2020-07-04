import utilities.LSTMConverter as LSTMConverter

def test():

    lstmConverter = LSTMConverter.LSTMConverter(cuda=True, max_layers=20)

    directions = [(1,(1,0,0,0)), (3,(4,0,0,0)),(19,(1,0,0,0)),(4,(0,1,0,0)),(10,(0,0,2))]

    for direction in directions:
        print("original direction: ", direction)
        value = lstmConverter.directionToTensor(direction=direction)
        print("tensor: ", value)
        print("tensor size: ", value.size())
        print("converting tensor to direction: ", lstmConverter.tensorToDirection(tensor=value))