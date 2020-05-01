import children.pytorch.Network as nw
import children.Interfaces as Inter
import children.pytorch.Functions as Functions
import children.pytorch.MutateNetwork as MutateNetwork
from DAO import GeneratorFromImage, GeneratorFromCIFAR

import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, objects):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 32, 32)
        self.fc1 = nn.Linear(16 * 1 * 1, 10)

    def forward(self, x):

        #lenght = len(self.conv1.weight[0].view(-1))
        #print("len1=", lenght)
        x = self.conv1(x) 

        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)

        x = x.view(-1, 16 * 1 * 1)

        x = self.fc1(x)

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

def Test_pytorchNetwork():
    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  10)
    dataGen.dataConv2d()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net([]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
    running_loss = 0.0

    for j in range(24000):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(dataGen.data[0].cuda())
        loss = criterion(outputs, dataGen.data[1].cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        dataGen.update()

        if j % 100 == 0:    # print every 2000 mini-batches
            print("Energy: ", running_loss, "i = ", j+1)
        running_loss = 0.0


                
def Test_Mutacion():

    print("creating networks")
    #(0, ks[i], len(dataGen.data[0]), 1, 1),

    
    #networkADN = ((0, 3, 5, 8, 8), (0, 5, 3, 4, 4), (1, 3, 2), (2,))
    #mutationADN = ((0, 3, 10, 2, 2), (0, 10, 20, 10, 10), (1, 20, 2), (2,))
    #mutationADN = ((0, 3, 5, 8, 8), (0, 5, 3, 3, 3), (0, 3, 3, 2, 2), (1, 3, 2), (2,))
    #networkADN = ((0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, 32, 32), (1, 5, 10), (2,))
    networkADN = ((0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0, 11, 5, 32, 32), (1, 5, 10), (2,))
    network = nw.Network(networkADN, cudaFlag=True)

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  50)
    #dataGen = GeneratorFromImage.GeneratorFromImage(2, 200)
    dataGen.dataConv2d()

    for epoch in range(24000):
        
        network.Training(data=dataGen, p=1, dt=0.01, labels=None)

        if epoch % 200 == 199:
            print("Average Loss=", network.getAverageLoss(50), " - i= ", epoch+1)

    print("Accuracy=", network.generateEnergy(dataGen))


#Test_pytorchNetwork()

#Test_Batch(dataGen)
Test_Mutacion()

