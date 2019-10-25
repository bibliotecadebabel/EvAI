import children.pytorch.Network as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
from DAO import GeneratorFromImage

import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, objects):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, objects[2], objects[0], objects[1])
        self.fc3 = nn.Linear(objects[2] * 1 * 1, 2)

    def forward(self, x):
        x = self.conv1(x)
        #print("output conv2d: ", x.shape)
        x = x.view(-1, 16 * 1 * 1)
        #print("input linear: ", x.shape)
        x = self.fc3(x)
        #print("output linear: ", x.shape)
        return x

def Test_node_3(network,n=100,dt=0.1):
    k=0
    layer_f=network.nodes[3].objects[0]
    #layer_i=network.nodes[2].objects[0]

    image = []
    image.append(network.nodes[0].objects[0].value)
    image.append(torch.tensor([0], dtype=torch.long))

    k=0
    A = network.nodes[2].objects[0].value
    A.requires_grad = True
    network.addFilters()
    while k < 200:
        
        #value = network.nodes[3].objects[0].object(A, image[1])
        network.assignLabels(image[1])
        network.nodes[2].objects[0].value.requires_grad = True
        Functions.Propagation(network.nodes[3].objects[0], 1)
        network.nodes[3].objects[0].value.backward()
        network.nodes[2].objects[0].value.requires_grad = False
        network.nodes[2].objects[0].value -= network.nodes[2].objects[0].value.grad * 0.1
        network.nodes[2].objects[0].value.grad.data.zero_()
        
        value = network.nodes[2].objects[0].value

        sc = value[0]
        sn = value[1]

        p = torch.exp(sc) / (torch.exp(sc) + torch.exp(sn))
        print(p)

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

def Test_realImage(network, dataGen):

    network.Training(data=dataGen.data, dt=10, p=2000)
    Inter.trakPytorch(network,'Net_folder_map', dataGen)

def Test_multipleNetworks(dataGen, x, y):
    networks = []
    size = dataGen.size
    ks = [2, 3, 4, 5, 6]


    for i in range(2):
        networks.append(nw.Network([x, y, ks[i]]))

    k = 0
    while True:
        for i in range(len(networks)):
            networks[i].Training(data=dataGen.data, dt=0.01, p=2)
        
        if k == 10:
            k = 0
            for i in range(len(networks)):
                print("Energy network of filters #", str(networks[i].objects[2]),": ", networks[i].total_value, " (", networks[i].Predict(dataGen.data[0]),")")  
        k +=1
    for j in range(len(networks)):
        networkName = "network-"+str(j)+"-map"
        Inter.trakPytorch(networks[i], networkName, dataGen)
    
def Test_CrossEntry(dataGen):

    net = nw.Network([dataGen.size[2], dataGen.size[3], 4])

    dataSub = []
    
    k = 0
    for data in dataGen.data:
        if k == 0:
            dataSub.append(data)
        k +=1
    #print("net: ",net.objects)
    net.Training(data=dataGen.data, dt=0.1, p=50000)
    #Inter.trakPytorch(net,'Net_folder_map', dataGen)
    #net.Predict(dataGen.data[10])
    #net.Predict(dataGen.data[0])
    #print("prob: ", net.getProbability())
    #net.Predict(dataGen.data[1])
    #print("prob: ", net.getProbability())


def Test_Batch(dataGen):

    batch = [dataGen.data]
    networks = []
    ks = [100, 153, 200, 5, 6]
    x = dataGen.size[1]
    y = dataGen.size[2]

    for i in range(1):
        print("creating networks")
        networkADN = ((0, 3, ks[i], x, y), (1, ks[i], 2), (2,))
        networks.append(nw.Network(networkADN, [x, y, ks[i]]))

    for _,a in enumerate(batch):
        print("Start Training")
        networks[0].Training(data=a[0], p=4000, dt=0.01, labels=a[1])
        Inter.trakPytorch(networks[0], "pokemon-netmap", dataGen)

def Test_pytorchNetwork(dataGen):
    batch = [dataGen.data]
    k = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net([dataGen.size[1], dataGen.size[2], k]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0
    for _,a in enumerate(batch):
        
        inputs, labels = a[0] / 255, a[1]

        for j in range(24000):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if j % 2000 == 1999:    # print every 2000 mini-batches
                print("Energy: ", running_loss, "i = ", j+1)
                running_loss = 0.0


dataGen = GeneratorFromImage.GeneratorFromImage(2, 100)
dataGen.dataConv2d()
size = dataGen.size


x = size[1]
y = size[2]
k = 2

#network = nw.Network([x,y,k])

Test_Batch(dataGen)

#Test_pytorchNetwork(dataGen)

#Test_node_3(network)
#Test_CrossEntry(dataGen)


#Test_multipleNetworks(dataGen, x, y)


#Test_realImage(network, dataGen)
#Test_modifyNetwork(network, dataGen.data)
#Test_node_2(network)

#Test_node_1(network)

#x = dataGen.data[0]

#print(x[0])

#print(x[0][0:2, 1:1])