import children.pytorch.Network as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from mutations.Dictionary import MutationsDictionary
import torch as torch

def executeMutation(oldNetwork, newAdn):
    
    network = nw.Network(newAdn, cudaFlag=oldNetwork.cudaFlag, momentum=oldNetwork.momentum)

    length_newadn = len(newAdn)
    length_oldadn = len(oldNetwork.adn)

    oldNetwork.updateGradFlag(False)
    network.updateGradFlag(False)

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

    oldNetwork.updateGradFlag(True)
    network.updateGradFlag(True)

    return network
