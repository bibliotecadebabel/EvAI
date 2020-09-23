import children.pytorch.Network as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from mutations.Dictionary import MutationsDictionary
import torch as torch

############### MUTACIONES ###############

def executeMutation(oldNetwork, newAdn):
    
    network = nw.Network(newAdn, cuda_flag=oldNetwork.cuda_flag, momentum=oldNetwork.momentum)

    length_newadn = len(newAdn)
    length_oldadn = len(oldNetwork.adn)

    oldNetwork.set_grad_flag(False)
    network.set_grad_flag(False)

    addLayer = False
    removeLayer = False

    if length_newadn > length_oldadn:
        addLayer = True
    elif length_oldadn > length_newadn:
        removeLayer = True

    if addLayer == False and removeLayer == False:

        __defaultMutationProcess(oldNetwork=oldNetwork, network=network, lenghtAdn=length_newadn)

    elif addLayer == True:

        __addLayerMutationProcess(oldNetwork=oldNetwork, network=network, lenghtOldAdn=length_oldadn)

    elif removeLayer == True:

        __removeLayerMutationProcess(oldNetwork=oldNetwork, network=network, lenghtNewAdn=length_newadn)

    oldNetwork.set_grad_flag(True)
    network.set_grad_flag(True)

    return network

def __defaultMutationProcess(oldNetwork, network, lenghtAdn):
    
    for i in range(1, lenghtAdn+1):

        oldLayer = oldNetwork.nodes[i].objects[0]
        newLayer = network.nodes[i].objects[0]

        if oldLayer.get_filters() is not None:
            __doMutate(oldLayer, newLayer, network.cuda_flag)
        
        if network.cuda_flag == True:
            torch.cuda.empty_cache()
        
def __addLayerMutationProcess(oldNetwork, network, lenghtOldAdn):

    newConv2d = network.nodes[1].objects[0]

    if newConv2d.adn[0] == 0: # Verify if newLayer is Conv2d
        __initNewConvolution(newConv2d)
    
    for i in range(1, lenghtOldAdn+1):

        oldLayer = oldNetwork.nodes[i].objects[0]
        newLayer = network.nodes[i+1].objects[0]

        if oldLayer.get_filters() is not None:
            __doMutate(oldLayer, newLayer, network.cuda_flag)
        
        if network.cuda_flag == True:
            torch.cuda.empty_cache()

def __removeLayerMutationProcess(oldNetwork, network, lenghtNewAdn):
    
    for i in range(1, lenghtNewAdn+1):

        oldLayer = oldNetwork.nodes[i+1].objects[0]
        newLayer = network.nodes[i].objects[0]

        if oldLayer.get_filters() is not None:
            __doMutate(oldLayer, newLayer, network.cuda_flag)
        
        if network.cuda_flag == True:
            torch.cuda.empty_cache()

def __doMutate(oldLayer, newLayer, cuda_flag):
    
    dictionaryMutation = MutationsDictionary()

    mutation = dictionaryMutation.getMutation(oldLayer.adn, newLayer.adn)
    
    oldBias = oldLayer.get_bias().clone()
    oldFilter = oldLayer.get_filters().clone()

    if mutation is not None:
        mutation.doMutate(oldFilter, oldBias, newLayer, cuda=cuda_flag)
    else:
        newLayer.set_filters(oldFilter)
        newLayer.set_bias(oldBias)

def __initNewConvolution(newConvolution):
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
    