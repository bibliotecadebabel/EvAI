import children.pytorch.NetworkDendrites as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from mutations.Dictionary import MutationsDictionary
import torch as torch

def executeMutation(oldNetwork, nodeTarget):
    
    newAdn = nodeTarget.objects[0].shape
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
        print("add layer mutation")
        index_layer = __getTargetLayerIndex(nodeTarget)
        __addLayerMutationProcess(oldNetwork=oldNetwork, network=network, lenghtOldAdn=length_oldadn, indexAdded=index_layer)

    elif length_oldadn > length_newadn: # remove layer
        print("remove layer mutation")
        index_layer = __getTargetLayerIndex(nodeTarget)
        __removeLayerMutationProcess(oldNetwork=oldNetwork, network=network, lengthNewAdn=length_newadn, indexRemoved=index_layer)

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

def __addLayerMutationProcess(oldNetwork, network, lenghtOldAdn, indexAdded):

    indexOldLayer = 0
    indexNewLayer = 0
    addedFound = False

    __initNewConvolution(network.nodes[indexAdded+1].objects[0])
    
    for i in range(1, lenghtOldAdn+1):

        indexNewLayer = i
        indexOldLayer = i

        if i == indexAdded + 1 or addedFound == True:
            addedFound = True
            indexNewLayer += 1

        oldLayer = oldNetwork.nodes[indexOldLayer].objects[0]
        newLayer = network.nodes[indexNewLayer].objects[0]

        print("from: ", oldLayer.adn)
        print("to: ", newLayer.adn)

        if oldLayer.getFilter() is not None:
            __doMutate(oldLayer, newLayer, network.cudaFlag)
        
        if network.cudaFlag == True:
            torch.cuda.empty_cache()

def __removeLayerMutationProcess(oldNetwork, network, lengthNewAdn, indexRemoved):

    indexOldLayer = 0
    indexNewLayer = 0
    removedFound = False

    for i in range(1, lengthNewAdn+1):
        
        indexNewLayer = i
        indexOldLayer = i

        if i == indexRemoved+1 or removedFound == True:
            removedFound = True
            indexOldLayer += 1

        oldLayer = oldNetwork.nodes[indexOldLayer].objects[0]
        newLayer = network.nodes[indexNewLayer].objects[0]

        print("from: ", oldLayer.adn)
        print("to: ", newLayer.adn)

        if oldLayer.getFilter() is not None:
            __doMutate(oldLayer, newLayer, network.cudaFlag)
        
        if network.cudaFlag == True:
            torch.cuda.empty_cache()

def __doMutate(oldLayer, newLayer, flagCuda):
    
    dictionaryMutation = MutationsDictionary()

    mutation_list = dictionaryMutation.getMutationList(oldLayer.adn, newLayer.adn)
    
    oldBias = oldLayer.getBias().clone()
    oldFilter = oldLayer.getFilter().clone()

    if mutation_list is not None:
        
        for mutation in mutation_list:
            print("mutation value: ", mutation.value)

            print("oldFilter")
            print(oldFilter.shape)
            mutation.doMutate(oldFilter, oldBias, newLayer, cuda=flagCuda)
            oldFilter = newLayer.getFilter()
            oldBias = newLayer.getBias()
            print("oldFitler mutated")
            print(oldFilter.shape)
    else:
        newLayer.setFilter(oldFilter)
        newLayer.setBias(oldBias)

def __initNewConvolution(newConvolution):
    factor_n = 0.25
    entries = newConvolution.adn[1]

    torch.nn.init.constant_(newConvolution.object.weight, factor_n / entries)
    torch.nn.init.constant_(newConvolution.object.bias, 0)

def __getTargetLayerIndex(node):

    direction = node.objects[0].objects[0].direction

    return direction[0]