import children.pytorch.Network as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from mutations.Dictionary import MutationsDictionary
import torch as torch

############### MUTACIONES ###############

def executeMutation(oldNetwork, newAdn):

    network = nw.Network(newAdn, cudaFlag=oldNetwork.cudaFlag)

    length_newadn = len(newAdn)
    length_oldadn = len(oldNetwork.adn)

    oldNetwork.updateGradFlag(False)
    network.updateGradFlag(False)

    addLayer = False
    RemoveLayer = False

    if length_newadn > length_oldadn:
        addLayer = True
    elif length_oldadn > length_newadn:
        RemoveLayer = True
    

    for i in range(1, length_newadn+1):

        if addLayer == False and RemoveLayer == False:

            oldLayer = oldNetwork.nodes[i].objects[0]
            newLayer = network.nodes[i].objects[0]

            if oldLayer.getFilter() is not None:
                traditionalMutation(oldLayer, newLayer, network.cudaFlag)

        elif addLayer == True:
            
            if i < length_newadn:
                oldLayer = oldNetwork.nodes[i].objects[0]

            newLayer = network.nodes[i].objects[0]

            if oldLayer is not None and oldLayer.getFilter() is not None:

                if oldLayer.adn[0] == newLayer.adn[0]:
                    traditionalMutation(oldLayer, newLayer, network.cudaFlag)

                else:
                    initNewConvolution(newLayer)
                    newLinear = network.nodes[i+1].objects[0]
                    traditionalMutation(oldLayer, newLinear, network.cudaFlag)
            

        if network.cudaFlag == True:
            torch.cuda.empty_cache()

    oldNetwork.updateGradFlag(True)
    network.updateGradFlag(True)


    #showParameters(network)
    return network


def traditionalMutation(oldLayer, newLayer, flagCuda):
    
    dictionaryMutation = MutationsDictionary()

    mutation = dictionaryMutation.getMutation(oldLayer.adn, newLayer.adn)
    
    oldBias = oldLayer.getBias().clone()
    oldFilter = oldLayer.getFilter().clone()

    if mutation is not None:
        #print("mutation=", mutation)
        mutation.doMutate(oldFilter, oldBias, newLayer, cuda=flagCuda)
    else:
        #print("sending parameters=", newLayer.adn)
        newLayer.setFilter(oldFilter)
        newLayer.setBias(oldBias)

def initNewConvolution(newConvolution):
    factor_n = 0.25
    entries = newConvolution.adn[1]

    torch.nn.init.constant_(newConvolution.object.weight, factor_n / entries)
    torch.nn.init.constant_(newConvolution.object.bias, 0)

    '''
    print("new conv2d filter")
    print(newConvolution.object.weight)
    print("new conv2d bias")
    print(newConvolution.object.bias)
    '''