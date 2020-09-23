import children.pytorch.Network as nw
import children.pytorch.Functions as Functions

import torch
import torch.nn as nn
import torch.tensor as tensor


def generateData(data, objects, n):

        circulo = []
        circulo.append(torch.zeros([1, objects[0], objects[1], 3], dtype=torch.float32))
        circulo.append(generateCircle())

        data.append(circulo)

        for i in range(objects[0]//2):
            for j in range(objects[1]):
                #circulo[0][i][j] = [255, 255, 255]
                circulo[0][0][i][j] = tensor([255,255, 255], dtype=torch.float32)

        for i in range(objects[0]//2, objects[0]):
            for j in range(objects[1]):
                #circulo[0][i][j] = [1,1,1]
                circulo[0][0][i][j] = tensor([1,1, 1], dtype=torch.float32)

        circulo[0].transpose_(1, 3)
        circulo[0].transpose_(2, 3)
    
        for i in range(n-1):
            imagenRandom = []
            imagenRandom.append(generateImageRandom(objects))
            imagenRandom.append(generateNotCircle())

            data.append(imagenRandom)
        
        #for elemnt in data:
        #    elemnt[0] = elemnt[0] / 255


def generateImageRandom(objects):
    image = torch.zeros([1, objects[0], objects[1], 3], dtype=torch.float32)


    for i in range(objects[0]):
        for j in range(objects[1]):
            image[0,i,j] = tensor([Functions.random.randint(1, 255),
                Functions.random.randint(1, 255), 
                Functions.random.randint(1, 255)], dtype=torch.float32)

    image.transpose_(1, 3)
    image.transpose_(2, 3)
    return image

def generateCircle():
        return tensor([0,1], dtype=torch.float32)
    
def generateNotCircle():
        return tensor([1,0], dtype=torch.float32)

def Test_node_3(network,n=100,dt=0.001):
    k=0
    layer_f=network.nodes[3].objects[0]
    #layer_i=network.nodes[2].objects[0]


    
    while k<50000:
        network.Reset_der_total()
        #layer_f.propagate(layer_f)
        Functions.Propagation(layer_f, 20)
        layer_f.value.backward()
        network.Regularize_der()
        network.Acumulate_der(n)
        #layer_i.value+=-layer_i.value_der*dt
        network.Update(dt)
        print("value of layer_f: ", layer_f.value)
        k=k+1
    

def Test_node_2(network,label="c",n=5,dt=0.001):
    k=0
    layer_f=network.nodes[4].objects[0]
    layer_i=network.nodes[2].objects[0]
    layer_i.label=label
    while k<5:
        network.Reset_der_total()
        j=3
        while j<5:
            layer=network.nodes[j].objects[0]
            layer.propagate(layer)
            j=j+1
        layer_i.backPropagate(layer_i)
        network.Acumulate_der(n)
        layer_i.value+=-layer_i.value_der*dt
        #network.Update(dt)
        print("value of layer_f: ", layer_f.value)
        k=k+1

def Test_node_1(network,n=5,dt=0.1):
    k=0
    layer_f=network.nodes[4].objects[0]
    layer_i=network.nodes[1].objects[0]
    while k<5:
        network.Reset_der_total()
        j=2
        while j<5:
            layer=network.nodes[j].objects[0]
            layer.propagate(layer)
            j=j+1
        layer_i.backPropagate(layer_i)
        network.Acumulate_der(1)
        network.Update(dt)
        print("value of layer_f: ", layer_f.value)
        k=k+1

def Test_node_0(network,n=1000,dt=0.1):
    k=0
    layer_f=network.nodes[4].objects[0]
    layer_i=network.nodes[0].objects[0]
    while k<n:
        network.Predict(layer_i.value)
        network.assign_labels("n")
        network.Reset_der_total()
        #j=1
        #while j<5:
            #layer=network.nodes[j].objects[0]
            #layer.propagate(layer)
            #j=j+1

        #layer_i.backPropagate(layer_i)
        Functions.Propagation(layer_f)
        Functions.BackPropagation(layer_i)
        network.Acumulate_der(1)
        network.Update(dt)
        #print("value of layer_f: ", layer_f.value)
        #print("value of layer_f: ", layer_f.node.parents[0].objects[0].value)
        #print("value_der nodo 3: ", network.nodes[3].objects[0].value_der)
        k=k+1

def Test_modifyNetwork(network, data):

    print("Entrenando red \n")
    network.Training(data=data, dt=0.001, p=1000)


x = 4
y = 5
k = 12



objects = Functions.np.full((3), (x, y, k))

network = nw.Network([x,y,k])

data = []

generateData(data, objects, 100)

Test_modifyNetwork(network, data)

#Test_node_3(network)
"""

print('testing node 3')
Test_node_3(network)
print('testing node 2 label=c')
Test_node_2(network)
print('testing node 2 label=n')
Test_node_2(network,"n")
print('testing node 1')
Test_node_1(network)
print('testing node 1')

"""
#Test_node_0(network)






#network.Training(data=data, dt=0.05, p=10)
