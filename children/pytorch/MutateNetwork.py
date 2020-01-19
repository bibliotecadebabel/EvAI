import children.pytorch.Network as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from mutations.Dictionary import MutationsDictionary

############### MUTACIONES ###############

def executeMutation(oldNetwork, newAdn):

    network = nw.Network(newAdn, cudaFlag=oldNetwork.cudaFlag)

    length_newadn = len(newAdn)
    length_oldadn = len(oldNetwork.adn)

    dictionaryMutation = MutationsDictionary()

    oldNetwork.updateGradFlag(False)
    network.updateGradFlag(False)

    if length_newadn == length_oldadn:
    
        for i in range(length_oldadn):

            oldLayer = oldNetwork.nodes[i+1].objects[0]
            newLayer = network.nodes[i+1].objects[0]

            if oldLayer.getFilter() is not None:
                
                mutation = dictionaryMutation.getMutation(oldLayer.adn, newLayer.adn)
                
                oldBias = oldLayer.getBias().clone()
                oldFilter = oldLayer.getFilter().clone()

                if mutation is not None:
                    mutation.doMutate(oldFilter, oldBias, newLayer)
                else:
                    newLayer.setFilter(oldFilter)
                    newLayer.setBias(oldBias)


    else:

        print("add or remove layer mutations")

    oldNetwork.updateGradFlag(True)
    network.updateGradFlag(True)

    #showParameters(network)
    return network


def showParameters(network):
    
    for node in network.nodes:

        layer = node.objects[0]
        print("adn layer: ", layer.adn)
        
        if layer.getFilter() is not None:
            print("Filter shape: ", layer.getFilter().shape)
            print("Bias shape: ", layer.getBias().shape)