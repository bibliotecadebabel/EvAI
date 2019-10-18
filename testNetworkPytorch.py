import children.pytorch.Network as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from DAO import GeneratorFromImage

import torch
import torch.nn as nn
import torch.tensor as tensor

def generateCircle():
        return tensor([0,1], dtype=torch.float32)
    
def generateNotCircle():
        return tensor([1,0], dtype=torch.float32)

def Test_node_3(network,n=100,dt=0.1):
    k=0
    layer_f=network.nodes[3].objects[0]
    #layer_i=network.nodes[2].objects[0]

    image = []
    image.append(network.nodes[0].objects[0].value)
    image.append(torch.tensor([1,0], dtype=torch.float32))

    k=0
    A = network.nodes[2].objects[0].value
    A.requires_grad = True
    network.addFilters()
    while k < 1000:
        
        #value = network.nodes[3].objects[0].object(A, image[1])
        network.assignLabels(image[1])
        network.nodes[2].objects[0].value.requires_grad = True
        Functions.Propagation(network.nodes[3].objects[0], 1)
        network.nodes[3].objects[0].value.backward()
        network.nodes[2].objects[0].value.requires_grad = False
        network.nodes[2].objects[0].value -= network.nodes[2].objects[0].value.grad * 0.1
        network.nodes[2].objects[0].value.grad.data.zero_()
        
        print(network.nodes[2].objects[0].value)

def Test_node_2(network,n=100,dt=0.1):
    k=0
    layer_f=network.nodes[3].objects[0]
    #layer_i=network.nodes[2].objects[0]

    #network.addFilters()

    image = []
    image.append(network.nodes[0].objects[0].value)
    image.append(torch.tensor([1,0], dtype=torch.float32))

    k=0

    A = network.nodes[2].objects[0].value
    A.requires_grad = True
    
    #print(network.nodes[2].objects[0].value)

    network.addFilters()
    while k < 1000:
        
        print("iter: ", k)
        network.updateGradFlag(True)
        network.nodes[1].objects[0].value.requires_grad = True
        network.assignLabels(image[1])
        network.nodes[2].objects[0].propagate(network.nodes[2].objects[0])
        network.nodes[3].objects[0].propagate(network.nodes[3].objects[0])
        network.nodes[3].objects[0].value.backward()
        network.updateGradFlag(False)
        network.nodes[1].objects[0].value.requires_grad = False
        #network.Regularize_der()
        network.Acumulate_der(1)
        network.Update(dt)

        #network.nodes[1].objects[0].value -= network.nodes[1].objects[0].value.grad * dt 

        network.Reset_der_total()
        network.Reset_der()

        print("Value Layer Linear:", network.nodes[2].objects[0].value)
        k+=1

def Test_node_1(network,n=100,dt=0.1):
    k=0
    layer_f=network.nodes[3].objects[0]
    #layer_i=network.nodes[2].objects[0]

    image = []
    image.append(network.nodes[0].objects[0].value)
    image.append(torch.tensor([1,0], dtype=torch.float32))

    k=0

    A = network.nodes[2].objects[0].value
    A.requires_grad = True
    
    print(network.nodes[2].objects[0].value)
    while k < 10000:
        
        network.updateGradFlag(True)
        network.assignLabels(image[1])
        network.nodes[1].objects[0].propagate(network.nodes[1].objects[0])
        network.nodes[2].objects[0].propagate(network.nodes[2].objects[0])
        network.nodes[3].objects[0].propagate(network.nodes[3].objects[0])
        network.nodes[3].objects[0].value.backward()
        network.updateGradFlag(False)
        #network.Regularize_der()
        network.Acumulate_der(1)
        network.Update(dt)
        network.Reset_der_total()
        network.Reset_der()
        network.Predict(image)
        print(network.getProbability())
        k+=1

def Test_modifyNetwork(network, data):

    for i in range(10000):
        #print("Entrenando red")
        print("Sin mutaciones")
        network.Training(data=data, dt=10, p=10)
        print("Agregando Filtro")
        network.addFilters()
        #print("Entrenando Red modificada")
        network.Training(data=data, dt=10, p=10)
        print("Eliminando Filtro")
        network.removeFilter()
        #print("Entrenando Red modificada")
        network.Training(data=data, dt=10, p=10)

def Test_realImage(network, dataGen):

    network.Training(data=dataGen.data, dt=0.1, p=2000)
    Inter.trakPytorch(network,'Net_folder_map', dataGen)



dataGen = GeneratorFromImage.GeneratorFromImage(2, 100)
dataGen.dataConv2d()
size = dataGen.data[0][0].shape


x = size[2]
y = size[3]
k = 100

network = nw.Network([x,y,k])

#Test_realImage(network, dataGen)
Test_modifyNetwork(network, dataGen.data)
#Test_node_2(network)

#Test_node_1(network)

#x = dataGen.data[0]

#print(x[0])

#print(x[0][0:2, 1:1])