import children.pytorch.NetworkDendrites as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from mutations.Dictionary import MutationsDictionary
import DNA_directions_f as direction_dna
import torch as torch
import mutations.Convolution2d.Mutations as Conv2dMutations
import const.mutation_type as m_type

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
        print("add layer mutation")
        index_layer = __getTargetIndex(oldAdn=oldNetwork.adn, newAdn=newAdn, direction_function=direction_dna.add_layer)
        __addLayerMutationProcess(oldNetwork=oldNetwork, network=network, lenghtOldAdn=length_oldadn, indexAdded=index_layer)

    elif length_oldadn > length_newadn: # remove layer
        print("remove layer mutation")
        index_layer = __getTargetIndex(oldAdn=oldNetwork.adn, newAdn=newAdn, direction_function=direction_dna.remove_layer)
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
    
    mutation_type, index_target = __getMutationTypeAndTargetIndex(oldAdn=oldNetwork.adn, newAdn=network.adn)
    source_dendrites = []

    if index_target is not None:
        source_dendrites = __getSourceLayerDendrites(indexLayer=index_target, oldAdn=oldNetwork.adn)

    print("source dendrites=", source_dendrites)

    for i in range(1, lenghtAdn+1):

        oldLayer = oldNetwork.nodes[i].objects[0]
        newLayer = network.nodes[i].objects[0]

        if oldLayer.getFilter() is not None:

            adjustFilterMutation = __getAdjustFilterMutation(indexLayer=i, source_dendrites=source_dendrites,
                                                                network=oldNetwork, adjustLayer=oldLayer)

            oldFilter = oldLayer.getFilter()

            if adjustFilterMutation is not None:

                oldFilter = adjustFilterMutation.adjustEntryFilters(oldFilter=oldFilter, newFilter=newLayer.getFilter(), 
                                                            mutation_type=mutation_type)

            __doMutate(oldFilter=oldFilter, oldBias=oldLayer.getBias(),
                        newLayer=newLayer, flagCuda=network.cudaFlag, layerType=oldLayer.adn[0])
        
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

        if oldLayer.getFilter() is not None:
            __doMutate(oldFilter=oldLayer.getFilter(), oldBias=oldLayer.getBias(),
                        newLayer=newLayer, flagCuda=network.cudaFlag, layerType=oldLayer.adn[0])
        
        if network.cudaFlag == True:
            torch.cuda.empty_cache()

def __removeLayerMutationProcess(oldNetwork, network, lengthNewAdn, indexRemoved):

    indexOldLayer = 0
    indexNewLayer = 0
    removedFound = False

    source_dendrites = __getSourceLayerDendrites(indexLayer=indexRemoved, oldAdn=oldNetwork.adn)

    print("source dendrites=", source_dendrites)
    for i in range(1, lengthNewAdn+1):
        
        indexNewLayer = i
        indexOldLayer = i

        if i == indexRemoved+1 or removedFound == True:
            removedFound = True
            indexOldLayer += 1

        oldLayer = oldNetwork.nodes[indexOldLayer].objects[0]
        newLayer = network.nodes[indexNewLayer].objects[0]

        if oldLayer.getFilter() is not None:

            adjustFilterMutation = __getAdjustFilterMutation(indexLayer=indexOldLayer, source_dendrites=source_dendrites,
                                                                network=oldNetwork, adjustLayer=oldLayer)
            
            oldFilter = oldLayer.getFilter()
            if adjustFilterMutation is not None:
                oldFilter = adjustFilterMutation.removeFilters()

            __doMutate(oldFilter=oldFilter, oldBias=oldLayer.getBias(),
                        newLayer=newLayer, flagCuda=network.cudaFlag, layerType=oldLayer.adn[0])
        
        if network.cudaFlag == True:
            torch.cuda.empty_cache()

def __doMutate(oldFilter, oldBias, layerType, newLayer, flagCuda):
    
    oldBias = oldBias.clone()
    oldFilter = oldFilter.clone()

    print("from: ", oldFilter.shape)
    print("to: ", newLayer.getFilter().shape)

    dictionaryMutation = MutationsDictionary()

    mutation_list = dictionaryMutation.getMutationList(layerType=layerType, 
        oldFilter=oldFilter, newFilter=newLayer.getFilter())
    

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
        print("default parameter sending")
        newLayer.setFilter(oldFilter)
        newLayer.setBias(oldBias)

def __initNewConvolution(newConvolution):
    factor_n = 0.25
    entries = newConvolution.adn[1]

    torch.nn.init.constant_(newConvolution.object.weight, factor_n / entries)
    torch.nn.init.constant_(newConvolution.object.bias, 0)

def __getMutationTypeAndTargetIndex(oldAdn, newAdn):

    same_dendrites = True
    mutation_type = None
    target_index = None
    
    for i in range(len(oldAdn)):
        
        if oldAdn[i][0] == 0:

            if oldAdn[i][2] > newAdn[i][2]:
                mutation_type = m_type.DEFAULT_REMOVE_FILTERS
                target_index = i - 1
                break
            
            elif oldAdn[i][2] < newAdn[i][2]:
                mutation_type = m_type.DEFAULT_ADD_FILTERS
                target_index = i - 1
                break

        if oldAdn[i][0] == 3:

            if oldAdn[i] != newAdn[i]:
                same_dendrites = False
                break
    
    if same_dendrites == False:
        mutation_type = m_type.DEFAULT_CHANGE_DENDRITES
        target_index = None

    
    print("mutation type=", mutation_type)
    print("target index=", target_index)
    return [mutation_type, target_index]
            


def __getTargetIndex(oldAdn, newAdn, direction_function):

    targetLayer = None
    indexConv2d = 0
    for i in range(len(oldAdn)):

        if oldAdn[i][0] == 0: #check if is conv2d
            generated_dna = direction_function(indexConv2d, oldAdn)
            if str(generated_dna) == str(newAdn):  
                targetLayer = indexConv2d
                break
            indexConv2d += 1
    
    return targetLayer

def __getSourceLayerDendrites(indexLayer, oldAdn):
    source_dendrites = []
    
    for layer in oldAdn:

        if layer[0] == 3: #check if is dendrite

            if layer[1] == indexLayer:
                source_dendrites.append(layer)
    
    return source_dendrites

def __getTargetLayerDendrites(indexLayer, adn):
    target_dendrites = []
    
    for layer in adn:

        if layer[0] == 3: #check if is dendrite

            if layer[2] == indexLayer:
                target_dendrites.append(layer)
    
    return target_dendrites

def __getSourceDendritesIndexLayers(indexLayer, source_dendrites, adn): 
    dendrite_affected = None
    index_layers = []

    for dendrite in source_dendrites:

        if dendrite[2] == indexLayer:
            dendrite_affected = dendrite
    
    if dendrite_affected is not None:

        target_dendrites = __getTargetLayerDendrites(indexLayer=dendrite_affected[2], adn=adn)

        for dendrite in target_dendrites:
            index_layers.append(dendrite[1])
    
    return index_layers

def __getAdjustFilterMutation(indexLayer, source_dendrites, network, adjustLayer):

    index_adn_list = __getSourceDendritesIndexLayers(indexLayer=indexLayer-1, source_dendrites=source_dendrites, adn=network.adn)
    mutation = None

    if len(index_adn_list) > 0:
        
        mutation = Conv2dMutations.AdjustEntryFilters_Dendrite(adjustLayer=adjustLayer, indexList=index_adn_list,
             targetIndex=source_dendrites[0][1], network=network)

    return mutation




                


        









