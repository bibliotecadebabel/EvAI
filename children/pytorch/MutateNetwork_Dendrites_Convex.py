import children.pytorch.NetworkDendrites as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from mutations.Dictionary import MutationsDictionary
import DNA_directions_convex as direction_dna
import torch as torch
import mutations.Convolution2d.Mutations as Conv2dMutations
import mutations.BatchNormalization.BatchNormalization as batchMutate
import const.mutation_type as m_type
import utilities.FileManager as FileManager

def executeMutation(oldNetwork, newAdn):
    fileManager = FileManager.FileManager()
    fileManager.setFileName("dnas_mutation_error.txt")
    try:
        network = nw.Network(newAdn, cudaFlag=oldNetwork.cudaFlag, momentum=oldNetwork.momentum,
                                weight_decay=oldNetwork.weight_decay, enable_activation=oldNetwork.enable_activation,
                                enable_track_stats=oldNetwork.enable_track_stats, dropout_value=oldNetwork.dropout_value,
                                dropout_function=oldNetwork.dropout_function, enable_last_activation=oldNetwork.enable_last_activation,
                                version=oldNetwork.version, eps_batchnorm=oldNetwork.eps_batchnorm)

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
            index_layer, mutation_type = __getTargetIndex(oldAdn=oldNetwork.adn, newAdn=newAdn)

            __addLayerMutationProcess(oldNetwork=oldNetwork, network=network, lenghtOldAdn=length_oldadn,
                                            indexAdded=index_layer, mutation_type=mutation_type)

        elif length_oldadn > length_newadn: # remove layer
            #print("remove layer mutation")
            index_layer = __getTargetRemoved(oldAdn=oldNetwork.adn, newAdn=newAdn)
            __removeLayerMutationProcess(oldNetwork=oldNetwork, network=network, lengthNewAdn=length_newadn, indexRemoved=index_layer)

    except:
        fileManager.appendFile("## MUTATION ##")
        fileManager.appendFile("old DNA: "+str(oldNetwork.adn))
        fileManager.appendFile("new DNA: "+str(newAdn))

        print("#### ERROR DNAs  ####")
        print("OLD")
        print(oldNetwork.adn)
        print("NEW")
        print(newAdn)
        raise

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
                                                                network=oldNetwork, adjustLayer=oldLayer, newFilter=newLayer.getFilter(),
                                                                mutationType=mutation_type, newNetwork=network)

            oldFilter = oldLayer.getFilter()
            oldBias = oldLayer.getBias()

            if adjustFilterMutation is not None:

                if oldLayer.adn[0] == 0:
                    oldFilter, oldBias = adjustFilterMutation.adjustEntryFilters(mutation_type=mutation_type)

            h_value = 1
            if oldLayer.tensor_h is not None:
                h_value = oldLayer.tensor_h.data.clone()

            __doMutate(oldFilter=oldFilter, oldBias=oldBias, oldBatchnorm=oldLayer.getBatchNorm(),
                        newLayer=newLayer, flagCuda=network.cudaFlag, layerType=oldLayer.adn[0], oldH=h_value)

        if network.cudaFlag == True:
            torch.cuda.empty_cache()

def __addLayerMutationProcess(oldNetwork, network, lenghtOldAdn, indexAdded, mutation_type):

    indexOldLayer = 0
    indexNewLayer = 0
    addedFound = False

    if mutation_type == m_type.ADD_POOL_LAYER:
        __initNewPoolConvolution(network.nodes[indexAdded+1].objects[0])
    else:
        __initNewConvolution(network.nodes[indexAdded+1].objects[0])

    for i in range(1, lenghtOldAdn+1):

        indexNewLayer = i
        indexOldLayer = i

        if i == indexAdded + 1 or addedFound == True:
            addedFound = True
            indexNewLayer += 1

        oldLayer = oldNetwork.nodes[indexOldLayer].objects[0]
        newLayer = network.nodes[indexNewLayer].objects[0]

        h_value = 1
        if oldLayer.tensor_h is not None:
            h_value = oldLayer.tensor_h.data.clone()

        if oldLayer.getFilter() is not None:
            __doMutate(oldFilter=oldLayer.getFilter(), oldBias=oldLayer.getBias(), oldBatchnorm=oldLayer.getBatchNorm(),
                        newLayer=newLayer, flagCuda=network.cudaFlag, layerType=oldLayer.adn[0], oldH=h_value)

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
                                                                network=oldNetwork, adjustLayer=oldLayer, 
                                                                newFilter=newLayer.getFilter())

            oldFilter = oldLayer.getFilter()
            oldBias = oldLayer.getBias()

            if adjustFilterMutation is not None:

                if oldLayer.adn[0] == 0:
                    oldFilter, oldBias = adjustFilterMutation.removeFilters()

            h_value = 1
            if oldLayer.tensor_h is not None:
                h_value = oldLayer.tensor_h.data.clone()

            __doMutate(oldFilter=oldFilter, oldBias=oldBias, oldBatchnorm=oldLayer.getBatchNorm(),
                        newLayer=newLayer, flagCuda=network.cudaFlag, layerType=oldLayer.adn[0], oldH=h_value)

        if network.cudaFlag == True:
            torch.cuda.empty_cache()

def __doMutate(oldFilter, oldBias, oldBatchnorm, layerType,  newLayer, flagCuda, oldH):

    oldBias = oldBias.clone()
    oldFilter = oldFilter.clone()

    dictionaryMutation = MutationsDictionary()

    mutation_list = dictionaryMutation.getMutationList(layerType=layerType,
        oldFilter=oldFilter, newFilter=newLayer.getFilter())

    if mutation_list is not None:

        #print("mutation list: ", mutation_list)
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
    
    if newLayer.tensor_h is not None:
        newLayer.tensor_h.data = oldH

def __initNewPoolConvolution(newConvolution):

    torch.nn.init.constant_(newConvolution.object.weight, 0)
    torch.nn.init.constant_(newConvolution.object.bias, 0)

def __initNewConvolution(newConvolution):

    torch.nn.init.constant_(newConvolution.object.weight, 0)
    torch.nn.init.constant_(newConvolution.object.bias, 0)

    shape = newConvolution.object.weight.shape

    for k in range(shape[0]):
        newConvolution.object.weight[k][k][0][0] = 1


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



def __getTargetIndex(oldAdn, newAdn):

    targetLayer = None
    indexConv2d = 0
    stop = False
    iterations = 0
    mutation_type = None
    while not stop:

        indexConv2d = 0
        for i in range(len(oldAdn)):

            if oldAdn[i][0] == 0: #check if is conv2d or linear
                generated_dna = direction_dna.add_layer(indexConv2d, oldAdn)
                if str(generated_dna) == str(newAdn):
                    targetLayer = indexConv2d
                    mutation_type = m_type.ADD_LAYER
                    stop = True
                    break

            if oldAdn[i][0] == 0: #check if is conv2d or linear
                generated_dna = direction_dna.add_pool_layer(indexConv2d, oldAdn)
                if str(generated_dna) == str(newAdn):
                    targetLayer = indexConv2d
                    mutation_type = m_type.ADD_POOL_LAYER
                    stop = True
                    break

                indexConv2d += 1


        if iterations > 100:
            print("WARNING STUCK TRYING TO FINDING ADDED LAYER")
            iterations = 0

        iterations += 1

    return [targetLayer, mutation_type]

def __getTargetRemoved(oldAdn, newAdn):
    targetLayer = None
    indexConv2d = 0
    for i in range(len(oldAdn)):

        if oldAdn[i][0] == 0: #check if is conv2d
            generated_dna = direction_dna.remove_layer(indexConv2d, oldAdn)
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

def __getAdjustFilterMutation(indexLayer, source_dendrites, network, adjustLayer, newFilter, mutationType=None, newNetwork=None):

    index_adn_list = __getSourceDendritesIndexLayers(indexLayer=indexLayer-1, source_dendrites=source_dendrites, adn=network.adn)
    mutation = None

    if len(index_adn_list) > 0:
        
        if mutationType is not None and mutationType == m_type.DEFAULT_ADD_FILTERS:
            index_adn_list = __verifyConvexDendrites(newNetwork=newNetwork, newFilter=newFilter, index_list=index_adn_list, targetIndex=indexLayer-1)
        
        mutation = Conv2dMutations.AdjustEntryFilters_Dendrite(adjustLayer=adjustLayer, indexList=index_adn_list,
             targetIndex=source_dendrites[0][1], network=network, newFilter=newFilter)

    return mutation

def __verifyConvexDendrites(newNetwork, newFilter, index_list, targetIndex):

    max_filters = 0
    entries_channel = newFilter.shape[1]
    total_filters = 0
    
    index_target = -1

    index_list_adjusted = index_list
    for parent_layer_index in index_list:

        parent_layer = newNetwork.adn[parent_layer_index+1]

        total_filters += parent_layer[2]

        if parent_layer[2] >= max_filters:
            index_target = parent_layer_index
            max_filters = parent_layer[2]

    if entries_channel != total_filters:
        index_list_adjusted = [index_target]
    
    return index_list_adjusted

        

