import DNA_directions_clone as dire
import children.pytorch.MutateNetwork_Dendrites_clone as Mutate_Clone
import children.pytorch.NetworkDendrites as nw_dendrites
from DAO import GeneratorFromImage, GeneratorFromCIFAR

def add_layer_test(parentDNA, index_layer):

    return dire.add_layer(index_layer,parentDNA)


def test_clone():
    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  25)
    dataGen.dataConv2d()

    parentDNA=((-1, 1, 3, 32, 32), (0, 3, 5, 3, 3), (0, 8, 15, 4, 4), (0, 15, 5, 3, 3), (0, 23, 30, 4, 4), 
                (0, 30, 5, 3, 3), (0, 53, 15, 32, 32), (1, 15, 10), (2,), (3, -1, 0), (3, -1, 1), (3, 0, 1), 
                (3, 1, 2), (3, 1, 3), (3, -1, 3), (3, 2, 3), (3, 3, 4), (3, 3, 5), (3, 1, 5), (3, -1, 5), 
                (3, 4, 5), (3, 5, 6), (3, 6, 7))

    #mutate_DNA = add_layer_test(parentDNA, 0)

    network = nw_dendrites.Network(parentDNA, cudaFlag=True, momentum=0.9)
    network.loadModel("saved_models/cifar/25_cifar-duplicate-restarts_ver_nocosine_model_10")

    print("original network DNA= ", network.adn)
    network.generateEnergy(dataGen)
    print("Accuracy= ", network.getAcurracy())

    base_dt = 0.002
    min_dt = 0.0005

    network.TrainingCosineLR_Restarts(dataGenerator=dataGen, base_dt=base_dt, epochs=10, etamin=min_dt)
    network.generateEnergy(dataGen)
    print("Accuracy= ", network.getAcurracy())
        

test_clone()