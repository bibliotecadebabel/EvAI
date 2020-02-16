import children.pytorch.Network as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
import children.pytorch.MutateNetwork as MutateNetwork
from DAO import GeneratorFromImage

import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, objects):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 2, 2)
        self.conv2 = nn.Conv2d(50, 25, 2, 2)
        self.conv3 = nn.Conv2d(25, 12, 2, 2)
        self.fc1 = nn.Linear(12 * 1 * 1, 10)
        self.fc2 = nn.Linear(10 * 1 * 1, 2)

    def forward(self, x):

        #lenght = len(self.conv1.weight[0].view(-1))
        #print("len1=", lenght)
        x = self.conv1(x) 

        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)

        #lenght = len(self.conv2.weight[0].view(-1))
        #print("len2=", lenght)
        x = self.conv2(x) 
        
        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)
        
        #lenght = len(self.conv3.weight[0].view(-1))
        #print("len3=", lenght)
        x = self.conv3(x) 

        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)

        x = x.view(-1, 12 * 1 * 1)

        x = self.fc1(x)
        x = self.fc2(x)

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

def Test_pytorchNetwork(dataGen):
    batch = [dataGen.data]
    k = 50
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

def Test_Batch(dataGen):

    batch = [dataGen.data]
    print("len data: ", len(dataGen.data[0]))
    networks = []
    ks = [100]
    x = dataGen.size[1]
    y = dataGen.size[2]
    for i in range(1):
        print("creating networks")
        #(0, ks[i], len(dataGen.data[0]), 1, 1),
        networkADN = ((0, 3, ks[i], x, y), (1, ks[i], 2), (2,))
        networks.append(nw.Network(networkADN, cudaFlag=True))

    for _,a in enumerate(batch):
        print("Start Training")
        networks[0].Training(data=a[0], p=15000, dt=0.01, labels=a[1])
        #print("Loss Array: ", networks[0].getLossArray())
        Inter.trakPytorch(networks[0], "pokemon-netmap", dataGen)

def Test_Mutacion(dataGen):
    batch = [dataGen.data]
    print("len data: ", len(dataGen.data[0]))
    ks = [10]
    x = dataGen.size[1]
    y = dataGen.size[2]

    print("creating networks")

    #PRUEBA JAVIER: networkADN = ((0, 3, 5, 1,3, 3), (0,1, 5, 3,7,7), (0,1,5,15,3, 3), (1,5 , 2), (2,))
    networkADN = ((0, 3, 5, 1,3, 3), (0,1, 5, 5,9,9), (1,5 , 2), (2,))
    #mutationADN = ((0, 3, 20, 1, 2, 2), (0, 1, 8, 5, 10, 10), (0, 16, 5, 8, 1, 1), (1, 5, 2), (2,))
    network = nw.Network(networkADN, cudaFlag=True)
    for _,a in enumerate(batch):
        print("red original (network) k=", *network.adn)
        network.Training(data=a[0], p=10000, dt=0.1, labels=a[1])
        #print("mutando")
        #network2 = MutateNetwork.executeMutation(network, mutationADN)
        #print("entrando red mutada (network2): ", *network2.adn)
        #network2.Training(data=a[0],p=10000, dt=0.1, labels=a[1])
        #print("entrenando de nuevo red original (network)")
        #network.Training(data=a[0], p=10000, dt=0.1, labels=a[1])
        Inter.trakPytorch(network, "pokemon-netmap", dataGen)


dataGen = GeneratorFromImage.GeneratorFromImage(2, 15000, cuda=True)
dataGen.dataConv3d()
size = dataGen.size


x = size[1]
y = size[2]
k = 2

#Test_pytorchNetwork(dataGen)

#Test_Batch(dataGen)
Test_Mutacion(dataGen)

