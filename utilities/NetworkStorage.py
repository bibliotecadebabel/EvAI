import torch
import os
import utilities.ExperimentSettings
import children.pytorch.network_dendrites as nw

def load_network(fileName, settings : utilities.ExperimentSettings.ExperimentSettings, path=None):

    if path is None:
        path = os.path.join("saved_models","cifar", fileName)
    
    
    checkpoint = torch.load(path)

    if settings.momentum == None:
        momentum = checkpoint['momentum']
    else:
        momentum = settings.momentum
    
    if settings.weight_decay == None:
        weight_decay = checkpoint['weight_decay']
    else:
        weight_decay = settings.weight_decay

    if settings.enable_activation == None:
        enable_activation = checkpoint['enable_activation']
    else:
        enable_activation = settings.enable_activation
    
    if settings.enable_last_activation == None:
        enable_last_activation = checkpoint['enable_last_activation']
    else:
        enable_last_activation = settings.enable_last_activation
    
    if settings.dropout_value == None:
        dropout_value = checkpoint['dropout_value']
    else:
        dropout_value = settings.dropout_value
    
    if settings.enable_track_stats == None:
        enable_track_stats = checkpoint['enable_track_stats']
    else:
        enable_track_stats = settings.enable_track_stats

    if settings.version == None:
        version = checkpoint['version']
    else:
        version = settings.version
    
    if settings.eps_batchorm == None:
        eps_batchnorm = checkpoint['eps_batchnorm']
    else:
        eps_batchnorm = settings.eps_batchorm

    network = nw.Network(dna=checkpoint['dna'], cuda_flag=settings.cuda, momentum=momentum, weight_decay=weight_decay, 
                enable_activation=enable_activation, enable_last_activation=enable_last_activation, dropout_value=dropout_value,
                dropout_function=settings.dropout_function, enable_track_stats=enable_track_stats, version=version, eps_batchnorm=eps_batchnorm)
    
    network.load_parameters(checkpoint=checkpoint)
    return network
    
def saveNetwork(network, fileName, path=None):

    if path is None:
        path = os.path.join("saved_models","cifar", fileName)
    
    network.save_model(path)
