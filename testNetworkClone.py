import DNA_directions_clone as dire
import children.pytorch.MutateNetwork_Dendrites_clone as Mutate_Clone
import children.pytorch.NetworkDendrites as nw_dendrites
from DAO import GeneratorFromImage, GeneratorFromCIFAR

def add_layer_test(parentDNA, index_layer):

    return dire.add_layer(index_layer,parentDNA)


def test_clone():
    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  25)
    dataGen.dataConv2d()

    parentDNA=((-1, 1, 3, 32, 32), (0, 3, 10, 3, 3), (0, 13, 15, 4, 4), (0, 15, 5, 3, 3), 
                (0, 23, 30, 4, 4), (0, 30, 5, 3, 3), (0, 53, 15, 32, 32), (1, 15, 10), (2,), (3, -1, 0), 
                (3, -1, 1), (3, 0, 1), (3, 1, 2), (3, 1, 3), (3, -1, 3), (3, 2, 3), (3, 3, 4), (3, 3, 5), 
                (3, 1, 5), (3, -1, 5), (3, 4, 5), (3, 5, 6), (3, 6, 7))

    mutate_DNA = add_layer_test(parentDNA, 0)

    network = nw_dendrites.Network(parentDNA, cudaFlag=True, momentum=0.9)

    print("original network DNA= ", network.adn)
    base_dt = 0.002
    min_dt = 0.001

    network.TrainingCosineLR_Restarts(dataGenerator=dataGen, base_dt=base_dt, epochs=1, etamin=min_dt)
    network.generateEnergy(dataGen)
    print("Accuracy (original)= ", network.getAcurracy())
    
    network_2 = Mutate_Clone.executeMutation(oldNetwork=network, newAdn=mutate_DNA)
    print("mutate network DNA= ", network_2.adn)
    network_2.generateEnergy(dataGen)
    print("Accuracy after mutation= ", network_2.getAcurracy())
    
    network_2.TrainingCosineLR_Restarts(dataGenerator=dataGen, base_dt=base_dt, epochs=1, etamin=min_dt)
    network_2.generateEnergy(dataGen)
    print("Accuracy after training mutation=", network_2.getAcurracy())
        

test_clone()