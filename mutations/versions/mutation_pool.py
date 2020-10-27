import children.pytorch.network_dendrites as nw
from mutations.mutations_dictionary import MutationsDictionary
import Geometric.Directions.DNA_directions_pool as direction_dna
import torch
import mutations.layers.conv2d.mutations as Conv2dMutations
import mutations.layers.batch_normalization.mutations as batchMutate
import const.mutation_type as m_type
import children.pytorch.layers.learnable_layers.layer_learnable as learnable_layer
import children.pytorch.layers.learnable_layers.layer_conv2d as conv2d_layer

def execute_mutation(old_network, new_dna):
    
    network = nw.Network(new_dna, cuda_flag=old_network.cuda_flag, momentum=old_network.momentum, 
                            weight_decay=old_network.weight_decay, enable_activation=old_network.enable_activation,
                            enable_track_stats=old_network.enable_track_stats, dropout_value=old_network.dropout_value,
                            dropout_function=old_network.dropout_function, enable_last_activation=old_network.enable_last_activation,
                            version=old_network.version, eps_batchnorm=old_network.eps_batchnorm)
                            
    network.loss_history = old_network.loss_history[-200:]

    length_new_dna = __generateLenghtDNA(new_dna)
    length_old_dna = __generateLenghtDNA(old_network.dna)

    old_network.set_grad_flag(False)
    network.set_grad_flag(False)


    if length_new_dna == length_old_dna:
        #print("default mutation process")
        __init_mutation(old_network=old_network, network=network, lenghtDna=length_new_dna)

    elif length_new_dna > length_old_dna: # add layer
        #print("add layer mutation")
        index_layer = __getTargetIndex(old_dna=old_network.dna, new_dna=new_dna, direction_function=direction_dna.add_layer)

        if index_layer == None:
            index_layer = __getTargetIndex(old_dna=old_network.dna, new_dna=new_dna, direction_function=direction_dna.add_pool_layer)

        __init_add_layer_mutation(old_network=old_network, network=network, lenghtold_dna=length_old_dna, added_layer_index=index_layer)

    elif length_old_dna > length_new_dna: # remove layer
        #print("remove layer mutation")
        index_layer = __getTargetIndex(old_dna=old_network.dna, new_dna=new_dna, direction_function=direction_dna.remove_layer)
        __init_remove_layer_mutation(old_network=old_network, network=network, lengthnew_dna=length_new_dna, removed_layer_index=index_layer)

    old_network.set_grad_flag(True)
    network.set_grad_flag(True)

    return network

def __generateLenghtDNA(dna):

    value = 0
    for i in range(len(dna)):

        tupleBody = dna[i]

        if tupleBody[0] >= 0 and tupleBody[0] <= 2:
            value += 1
    
    return value

def __init_mutation(old_network, network, lenghtDna):
    
    mutation_type, index_target = __getMutationTypeAndTargetIndex(old_dna=old_network.dna, new_dna=network.dna)
    
    source_dendrites = []

    if mutation_type == m_type.DEFAULT_ADD_FILTERS or mutation_type == m_type.DEFAULT_REMOVE_FILTERS:
        source_dendrites = __getSourceLayerDendrites(indexLayer=index_target, old_dna=old_network.dna)
    elif mutation_type == m_type.DEFAULT_REMOVE_DENDRITE:
        source_dendrites = __getRemovedDendrite(old_dna=old_network.dna, new_dna=network.dna)

    #print("source dendrites=", source_dendrites)

    for i in range(1, lenghtDna+1):

        old_layer = old_network.nodes[i].objects[0]
        new_layer = network.nodes[i].objects[0]

        if isinstance(old_layer, learnable_layer.LearnableLayer) and isinstance(new_layer, learnable_layer.LearnableLayer) and old_layer.get_filters() is not None:

            adjustFilterMutation = __getAdjustFilterMutation(indexLayer=i, source_dendrites=source_dendrites,
                                                                network=old_network, adjustLayer=old_layer, newFilter=new_layer.get_filters())

            oldFilter = old_layer.get_filters()
            oldBias = old_layer.get_bias()

            if adjustFilterMutation is not None:
                
                if old_layer.dna[0] == 0:
                    oldFilter, oldBias = adjustFilterMutation.adjustEntryFilters(mutation_type=mutation_type)

            if isinstance(old_layer, conv2d_layer.Conv2dLayer):
                __execute_mutations(oldFilter=oldFilter, oldBias=oldBias, oldBatchnorm=old_layer.get_batch_norm(),
                        new_layer=new_layer, cuda_flag=network.cuda_flag, layerType=old_layer.dna[0])
            else:
                __execute_mutations(oldFilter=oldFilter, oldBias=oldBias, oldBatchnorm=None,
                        new_layer=new_layer, cuda_flag=network.cuda_flag, layerType=old_layer.dna[0])
        
        if network.cuda_flag == True:
            torch.cuda.empty_cache()

def __init_add_layer_mutation(old_network, network, lenghtold_dna, added_layer_index):

    old_layer_index = 0
    new_layer_index = 0
    addedFound = False

    __initNewConvolution(network.nodes[added_layer_index+1].objects[0])
    
    for i in range(1, lenghtold_dna+1):

        new_layer_index = i
        old_layer_index = i

        if i == added_layer_index + 1 or addedFound == True:
            addedFound = True
            new_layer_index += 1

        old_layer = old_network.nodes[old_layer_index].objects[0]
        new_layer = network.nodes[new_layer_index].objects[0]

        if old_layer.get_filters() is not None:
            __execute_mutations(oldFilter=old_layer.get_filters(), oldBias=old_layer.get_bias(), oldBatchnorm=old_layer.get_batch_norm(),
                        new_layer=new_layer, cuda_flag=network.cuda_flag, layerType=old_layer.dna[0])
        
        if network.cuda_flag == True:
            torch.cuda.empty_cache()

def __init_remove_layer_mutation(old_network, network, lengthnew_dna, removed_layer_index):

    old_layer_index = 0
    new_layer_index = 0
    removed_layer_found = False

    source_dendrites = __getSourceLayerDendrites(indexLayer=removed_layer_index, old_dna=old_network.dna)

    for i in range(1, lengthnew_dna+1):
        
        new_layer_index = i
        old_layer_index = i

        if i == removed_layer_index+1 or removed_layer_found == True:
            removed_layer_found = True
            old_layer_index += 1

        old_layer = old_network.nodes[old_layer_index].objects[0]
        new_layer = network.nodes[new_layer_index].objects[0]

        if old_layer.get_filters() is not None:

            adjustFilterMutation = __getAdjustFilterMutation(indexLayer=old_layer_index, source_dendrites=source_dendrites,
                                                                network=old_network, adjustLayer=old_layer, newFilter=new_layer.get_filters())
            
            oldFilter = old_layer.get_filters()
            oldBias = old_layer.get_bias()

            if adjustFilterMutation is not None:

                if old_layer.dna[0] == 0:
                    oldFilter, oldBias = adjustFilterMutation.removeFilters()

            __execute_mutations(oldFilter=oldFilter, oldBias=oldBias, oldBatchnorm=old_layer.get_batch_norm(),
                        new_layer=new_layer, cuda_flag=network.cuda_flag, layerType=old_layer.dna[0])
        
        if network.cuda_flag == True:
            torch.cuda.empty_cache()

def __execute_mutations(oldFilter, oldBias, oldBatchnorm, layerType,  new_layer, cuda_flag):
    
    oldBias = oldBias.clone()
    oldFilter = oldFilter.clone()

    dictionaryMutation = MutationsDictionary()

    mutation_list = dictionaryMutation.get_mutation_list(layerType=layerType, 
        oldFilter=oldFilter, newFilter=new_layer.get_filters())

    if mutation_list is not None:

        for mutation in mutation_list:

            mutation.execute(oldFilter, oldBias, new_layer, cuda=cuda_flag)
            oldFilter = new_layer.get_filters()
            oldBias = new_layer.get_bias()

        norm_mutation = batchMutate.BatchNormMutation()
        norm_mutation.execute(oldBatchNorm=oldBatchnorm, new_layer=new_layer)

    else:
        new_layer.set_filters(oldFilter)
        new_layer.set_bias(oldBias)

        if isinstance(new_layer, conv2d_layer.Conv2dLayer):
            new_layer.set_batch_norm(oldBatchnorm)

def __initNewConvolution(newConvolution):
    factor_n = 0.25
    entries = newConvolution.dna[1]

    torch.nn.init.constant_(newConvolution.object.weight, 0)
    torch.nn.init.constant_(newConvolution.object.bias, 0)

def __getMutationTypeAndTargetIndex(old_dna, new_dna):

    mutation_type = None
    target_index = None

    for i in range(len(old_dna)):
        
        if old_dna[i][0] == 0:

            if old_dna[i][2] > new_dna[i][2]:
                mutation_type = m_type.DEFAULT_REMOVE_FILTERS
                target_index = i - 1
                break
            
            elif old_dna[i][2] < new_dna[i][2]:
                mutation_type = m_type.DEFAULT_ADD_FILTERS
                target_index = i - 1
                break    
    
    if target_index == None:

        # Como no se alteraron las salidas, se verifica si existe un layer que se le redujeron las entradas
        target_index = __getIndexLayerAffectedRemovedDendrite(old_dna, new_dna)

        # Si se encontro un layer con menos entradas indica que una dendrita fue eliminada
        if target_index != None:
            mutation_type = m_type.DEFAULT_REMOVE_DENDRITE

    return [mutation_type, target_index]
            
def __getIndexLayerAffectedRemovedDendrite(old_dna, new_dna):

    index_target = None
    
    # Obtengo cual es el layer afectado por la dendrita eliminada
    for i in range(len(old_dna)):

        if old_dna[i][0] == 0:

            if old_dna[i][1] > new_dna[i][1]:
                index_target = i-1
                break

    return index_target

def __getRemovedDendrite(old_dna, new_dna):

    removed_dendrite = []
    for i in range(len(old_dna)):

        dendrite_found = False
        
        if old_dna[i][0] == 3:

            for j in range(len(new_dna)):

                if new_dna[j][0] == 3 and new_dna[j] == old_dna[i]:
                    dendrite_found = True
                    break

            if dendrite_found == False:
                removed_dendrite.append(old_dna[i])
                break

    return removed_dendrite
    


def __getTargetIndex(old_dna, new_dna, direction_function):

    targetLayer = None
    indexConv2d = 0
    for i in range(len(old_dna)):

        if old_dna[i][0] == 0: #check if is conv2d
            generated_dna = direction_function(indexConv2d, old_dna)
            if str(generated_dna) == str(new_dna):  
                targetLayer = indexConv2d
                break
            indexConv2d += 1
    
    return targetLayer

def __getSourceLayerDendrites(indexLayer, old_dna):
    source_dendrites = []
    
    for layer in old_dna:

        if layer[0] == 3: #check if is dendrite

            if layer[1] == indexLayer:
                source_dendrites.append(layer)
    
    return source_dendrites

def __getTargetLayerDendrites(indexLayer, dna):
    target_dendrites = []
    
    for layer in dna:

        if layer[0] == 3: #check if is dendrite

            if layer[2] == indexLayer:
                target_dendrites.append(layer)
    
    return target_dendrites

def __getSourceDendritesIndexLayers(indexLayer, source_dendrites, dna): 
    dendrite_affected = None
    index_layers = []

    for dendrite in source_dendrites:

        if dendrite[2] == indexLayer:
            dendrite_affected = dendrite
    
    if dendrite_affected is not None:

        target_dendrites = __getTargetLayerDendrites(indexLayer=dendrite_affected[2], dna=dna)

        for dendrite in target_dendrites:
            index_layers.append(dendrite[1])
    
    return index_layers

def __getAdjustFilterMutation(indexLayer, source_dendrites, network, adjustLayer, newFilter):

    index_dna_list = __getSourceDendritesIndexLayers(indexLayer=indexLayer-1, source_dendrites=source_dendrites, dna=network.dna)
    mutation = None

    if len(index_dna_list) > 0:
        
        mutation = Conv2dMutations.AdjustInputChannels(adjustLayer=adjustLayer, indexList=index_dna_list,
             targetIndex=source_dendrites[0][1], network=network, newFilter=newFilter)

    return mutation
    



                


        









