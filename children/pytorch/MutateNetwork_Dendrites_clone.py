import children.pytorch.NetworkDendrites as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from mutations.Dictionary import MutationsDictionary
import DNA_directions_pool as direction_dna
import torch as torch
import mutations.Convolution2d.Mutations as Conv2dMutations
import mutations.BatchNormalization.BatchNormalization as batchMutate
import const.mutation_type as m_type

def executeMutation(oldNetwork, newAdn):
    
    network = nw.Network(newAdn, cudaFlag=oldNetwork.cudaFlag, momentum=oldNetwork.momentum, 
                            weight_decay=oldNetwork.weight_decay, enable_activation=oldNetwork.enable_activation,
                            enable_track_stats=oldNetwork.enable_track_stats, dropout_value=oldNetwork.dropout_value,
                            dropout_function=oldNetwork.dropout_function)

    network.history_loss = oldNetwork.history_loss[-200:]

    length_newadn = __generateLenghtADN(newAdn)
    length_oldadn = __generateLenghtADN(oldNetwork.adn)

    oldNetwork.updateGradFlag(False)
    network.updateGradFlag(False)


    if length_newadn == length_oldadn:
        #print("default mutation process")
        __defaultMutationProcess(oldNetwork=oldNetwork, network=network, lenghtAdn=length_newadn)

    elif length_newadn > length_oldadn: # add layer
        #print("add layer mutation")
        index_layer = __getTargetIndex(oldAdn=oldNetwork.adn, newAdn=newAdn, direction_function=direction_dna.add_layer)

        if index_layer == None:
            index_layer = __getTargetIndex(oldAdn=oldNetwork.adn, newAdn=newAdn, direction_function=direction_dna.add_pool_layer)

        __addLayerMutationProcess(oldNetwork=oldNetwork, network=network, lenghtOldAdn=length_oldadn, indexAdded=index_layer)

    elif length_oldadn > length_newadn: # remove layer
        #print("remove layer mutation")
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

    if mutation_type == m_type.DEFAULT_ADD_FILTERS or mutation_type == m_type.DEFAULT_REMOVE_FILTERS:
        source_dendrites = __getSourceLayerDendrites(indexLayer=index_target, oldAdn=oldNetwork.adn)
    elif mutation_type == m_type.DEFAULT_REMOVE_DENDRITE:
        source_dendrites = __getRemovedDendrite(oldAdn=oldNetwork.adn, newAdn=network.adn)

    #print("source dendrites=", source_dendrites)

    for i in range(1, lenghtAdn+1):

        oldLayer = oldNetwork.nodes[i].objects[0]
        newLayer = network.nodes[i].objects[0]

        if oldLayer.getFilter() is not None:

            adjustFilterMutation = __getAdjustFilterMutation(indexLayer=i, source_dendrites=source_dendrites,
                                                                network=oldNetwork, adjustLayer=oldLayer, newFilter=newLayer.getFilter())

            oldFilter = oldLayer.getFilter()
            oldBias = oldLayer.getBias()

            if adjustFilterMutation is not None:
                
                if oldLayer.adn[0] == 0:
                    oldFilter, oldBias = adjustFilterMutation.adjustEntryFilters(mutation_type=mutation_type)

            __doMutate(oldFilter=oldFilter, oldBias=oldBias, oldBatchnorm=oldLayer.getBatchNorm(),
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
            __doMutate(oldFilter=oldLayer.getFilter(), oldBias=oldLayer.getBias(), oldBatchnorm=oldLayer.getBatchNorm(),
                        newLayer=newLayer, flagCuda=network.cudaFlag, layerType=oldLayer.adn[0])
        
        if network.cudaFlag == True:
            torch.cuda.empty_cache()

def __removeLayerMutationProcess(oldNetwork, network, lengthNewAdn, indexRemoved):

    indexOldLayer = 0
    indexNewLayer = 0
    removedFound = False

    source_dendrites = __getSourceLayerDendrites(indexLayer=indexRemoved, oldAdn=oldNetwork.adn)

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
                                                                network=oldNetwork, adjustLayer=oldLayer, newFilter=newLayer.getFilter())
            
            oldFilter = oldLayer.getFilter()
            oldBias = oldLayer.getBias()

            if adjustFilterMutation is not None:

                if oldLayer.adn[0] == 0:
                    oldFilter, oldBias = adjustFilterMutation.removeFilters()

            __doMutate(oldFilter=oldFilter, oldBias=oldBias, oldBatchnorm=oldLayer.getBatchNorm(),
                        newLayer=newLayer, flagCuda=network.cudaFlag, layerType=oldLayer.adn[0])
        
        if network.cudaFlag == True:
            torch.cuda.empty_cache()

def __doMutate(oldFilter, oldBias, oldBatchnorm, layerType,  newLayer, flagCuda):
    
    oldBias = oldBias.clone()
    oldFilter = oldFilter.clone()

    dictionaryMutation = MutationsDictionary()

    mutation_list = dictionaryMutation.getMutationList(layerType=layerType, 
        oldFilter=oldFilter, newFilter=newLayer.getFilter())

    if mutation_list is not None:

        for mutation in mutation_list:

            mutation.doMutate(oldFilter, oldBias, newLayer, cuda=flagCuda)
            oldFilter = newLayer.getFilter()
            oldBias = newLayer.getBias()

        norm_mutation = batchMutate.MutateBatchNormalization()
        norm_mutation.doMutate(oldBatchNorm=oldBatchnorm, newLayer=newLayer)

    else:
        newLayer.setFilter(oldFilter)
        newLayer.setBias(oldBias)
        newLayer.setBarchNorm(oldBatchnorm)

def __initNewConvolution(newConvolution):
    factor_n = 0.25
    entries = newConvolution.adn[1]

    torch.nn.init.constant_(newConvolution.object.weight, 0)
    torch.nn.init.constant_(newConvolution.object.bias, 0)

def __getMutationTypeAndTargetIndex(oldAdn, newAdn):

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
    
    if target_index == None:

        # Como no se alteraron las salidas, se verifica si existe un layer que se le redujeron las entradas
        target_index = __getIndexLayerAffectedRemovedDendrite(oldAdn, newAdn)

        # Si se encontro un layer con menos entradas indica que una dendrita fue eliminada
        if target_index != None:
            mutation_type = m_type.DEFAULT_REMOVE_DENDRITE

    return [mutation_type, target_index]
            
def __getIndexLayerAffectedRemovedDendrite(oldAdn, newAdn):

    index_target = None
    
    # Obtengo cual es el layer afectado por la dendrita eliminada
    for i in range(len(oldAdn)):

        if oldAdn[i][0] == 0:

            if oldAdn[i][1] > newAdn[i][1]:
                index_target = i-1
                break

    return index_target

def __getRemovedDendrite(oldAdn, newAdn):

    removed_dendrite = []
    for i in range(len(oldAdn)):

        dendrite_found = False
        
        if oldAdn[i][0] == 3:

            for j in range(len(newAdn)):

                if newAdn[j][0] == 3 and newAdn[j] == oldAdn[i]:
                    dendrite_found = True
                    break

            if dendrite_found == False:
                removed_dendrite.append(oldAdn[i])
                break

    return removed_dendrite
    


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

def __getAdjustFilterMutation(indexLayer, source_dendrites, network, adjustLayer, newFilter):

    index_adn_list = __getSourceDendritesIndexLayers(indexLayer=indexLayer-1, source_dendrites=source_dendrites, adn=network.adn)
    mutation = None

    if len(index_adn_list) > 0:
        
        mutation = Conv2dMutations.AdjustEntryFilters_Dendrite(adjustLayer=adjustLayer, indexList=index_adn_list,
             targetIndex=source_dendrites[0][1], network=network, newFilter=newFilter)

    return mutation
    



                


        









