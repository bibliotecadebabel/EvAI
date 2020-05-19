from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising, Damped_Alaising)
from DAO import GeneratorFromCIFAR
import children.pytorch.NetworkDendrites as nw

CUDA = True
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
BATCH_SIZE = 128
EPOCHS = 20


Alai=Damped_Alaising(Max_iter=3910)

ADN = ((-1,1,3,32,32), (0,3, 5, 3 , 3),(0, 5, 5, 3,  3), (0,5, 40, 32-4, 32-4),
        (1, 40, 10),(2,),(3,-1,0),(3,0,1),(3,1,2),(3,2,3),(3,3,4))


dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  BATCH_SIZE)
dataGen.dataConv2d()

network = nw.Network(adn=ADN, cudaFlag=CUDA, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

network.TrainingALaising(dataGenerator=dataGen, epochs=EPOCHS, alaising_object=Alai)
network.generateEnergy(dataGen)

print("ACCURACY= ", network.getAcurracy())