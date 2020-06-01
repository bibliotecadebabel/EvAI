import torch
import os
import TestNetwork.ExperimentSettings
import children.pytorch.NetworkDendrites as nw

def loadNetwork(fileName, settings : TestNetwork.ExperimentSettings.ExperimentSettings, path=None):

    if path is None:
        final_path = os.path.join("saved_models","cifar", fileName)
    
    checkpoint = torch.load(path)

    if settings.momentum == None:
        momentum = checkpoint['momentum']
    
    

    nw.Network(adn=checkpoint['adn'], cudaFlag=settings.cuda, momentum=settings.momentum, )



