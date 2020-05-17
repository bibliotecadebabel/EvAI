import DNA_directions_clone as dire
import children.pytorch.MutateNetwork_Dendrites_clone as Mutate_Clone
import children.pytorch.NetworkDendrites as nw_dendrites
from DAO import GeneratorFromImage, GeneratorFromCIFAR

def add_layer_test(parentDNA, index_layer):

    return dire.add_layer(index_layer,parentDNA)


def test_clone():

    base_dt = float(input("base_dt: "))
    min_dt = float(input("min_dt: "))

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  32)
    dataGen.dataConv2d()

    parentDNA=((-1, 1, 3, 32, 32), (0, 3, 15, 3, 3), (0, 18, 60, 3, 3), (0, 60, 30, 3, 3), 
            (0, 30, 30, 3, 3), (0, 120, 30, 4, 4), (0, 105, 30, 5, 5), (0, 138, 30, 32, 32), 
            (1, 30, 10), (2,), (3, -1, 0), (3, 0, 1), (3, -1, 1), (3, 1, 2), (3, 2, 3), (3, 1, 4), 
            (3, 2, 4), (3, 3, 4), (3, 1, 5), (3, 4, 5), (3, 0, 5), (3, 1, 6), (3, 0, 6), (3, -1, 6), 
            (3, 5, 6), (3, 4, 6), (3, 6, 7), (3, 7, 8))

    #mutate_DNA = add_layer_test(parentDNA, 0)

    network = nw_dendrites.Network(parentDNA, cudaFlag=True, momentum=0.9)
    network.loadModel("saved_models/cifar/3_cifar_test_restart-clone_1_model_20")

    print("original network DNA= ", network.adn)
    network.generateEnergy(dataGen)
    print("Accuracy= ", network.getAcurracy())

    network.TrainingCosineLR_Restarts(dataGenerator=dataGen, base_dt=base_dt, epochs=10, etamin=min_dt, weight_decay=0.0005)
    network.generateEnergy(dataGen)
    print("Accuracy= ", network.getAcurracy())
        

test_clone()