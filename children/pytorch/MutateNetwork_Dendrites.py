import children.pytorch.NetworkDendrites as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from mutations.Dictionary import MutationsDictionary
import torch as torch

def executeMutation(oldNetwork, newAdn):
    
    network = nw.Network(newAdn, cudaFlag=oldNetwork.cudaFlag, momentum=oldNetwork.momentum)

    length_newadn = __generateLenghtADN(newAdn)
    length_oldadn = __generateLenghtADN(oldNetwork.adn)

    print("len adn=", length_newadn)
    oldNetwork.updateGradFlag(False)
    network.updateGradFlag(False)


    if length_newadn == length_oldadn:
        print("default mutation process")
        __defaultMutationProcess(oldNetwork=oldNetwork, network=network, lenghtAdn=length_newadn)

    elif length_newadn > length_oldadn: # add layer
        pass

        #__addLayerMutationProcess(oldNetwork=oldNetwork, network=network, lenghtOldAdn=length_oldadn)

    elif length_oldadn > length_newadn: # remove layer
        pass
        #__removeLayerMutationProcess(oldNetwork=oldNetwork, network=network, lenghtNewAdn=length_newadn)

    oldNetwork.updateGradFlag(True)
    network.updateGradFlag(True)

    return network

def __generateLenghtADN(adn):

    value = 0
    for i in range(len(adn)):

        tupleBody = adn[i]

        if tupleBody[0] >= 0 and tupleBody[0] <= 2:
            value += 1
    
    return value

def __defaultMutationProcess(oldNetwork, network, lenghtAdn):
    
    for i in range(1, lenghtAdn+1):

        oldLayer = oldNetwork.nodes[i].objects[0]
        newLayer = network.nodes[i].objects[0]

        if oldLayer.getFilter() is not None:
            __doMutate(oldLayer, newLayer, network.cudaFlag)
        
        if network.cudaFlag == True:
            torch.cuda.empty_cache()

def __doMutate(oldLayer, newLayer, flagCuda):
    
    dictionaryMutation = MutationsDictionary()

    mutation = dictionaryMutation.getMutation(oldLayer.adn, newLayer.adn)
    
    oldBias = oldLayer.getBias().clone()
    oldFilter = oldLayer.getFilter().clone()

    if mutation is not None:
        mutation.doMutate(oldFilter, oldBias, newLayer, cuda=flagCuda)
    else:
        newLayer.setFilter(oldFilter)
        newLayer.setBias(oldBias)

def __initNewConvolution(newConvolution):
    factor_n = 0.25
    entries = newConvolution.adn[1]

    torch.nn.init.constant_(newConvolution.object.weight, factor_n / entries)
    torch.nn.init.constant_(newConvolution.object.bias, 0)