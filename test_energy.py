import children.net2.Network as nw
from utilities import Data_generator
import children.Interfaces as Inter
import decimal

import children.pytorch.Network as nw_p
import children.pytorch.MutateNetwork as MutateNetwork
from DAO import GeneratorFromImage

import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim

decimal.getcontext().prec = 100


class NetPytorch(nn.Module):
    def __init__(self, typeNet):
        super(NetPytorch, self).__init__()
        self.typeNet = typeNet
        if typeNet == 1:
            self.conv1 = nn.Conv2d(3, 1, (3, 3))
            self.conv2 = nn.Conv2d(4, 3, (11, 11))
            self.fc1 = nn.Linear(3 * 1 * 1, 2)
        else:
            self.conv1 = nn.Conv2d(3, 1, (3, 3))
            self.conv2 = nn.Conv2d(4, 1, (3, 3))
            self.conv3 = nn.Conv2d(4, 3, (11, 11))
            self.fc1 = nn.Linear(3 * 1 * 1, 2)

    def upgradeFlag(self, flag):

        self.conv1.weight.requires_grad = flag
        self.conv1.bias.requires_grad = flag

        if self.conv1.weight.grad is not None:
            self.conv1.weight.grad.requires_grad = flag
            self.conv1.bias.grad.requires_grad = flag

        self.conv2.weight.requires_grad = flag
        self.conv2.bias.requires_grad = flag

        if self.conv2.weight.grad is not None:
            self.conv2.weight.grad.requires_grad = flag
            self.conv2.bias.grad.requires_grad = flag

        self.fc1.weight.requires_grad = flag
        self.fc1.bias.requires_grad = flag

        if self.fc1.weight.grad is not None:
            self.fc1.weight.grad.requires_grad = flag
            self.fc1.bias.grad.requires_grad = flag

        if self.typeNet == 2:
            self.conv3.weight.requires_grad = flag
            self.conv3.bias.requires_grad = flag

            if self.conv3.weight.grad is not None:
                self.conv3.weight.grad.requires_grad = flag
                self.conv3.bias.grad.requires_grad = flag




    def recieveParameters(self, oldConv2d_1, oldConv2d_2, oldLinear):

        if self.typeNet == 2:
            print("recibiendo parametros")

            self.__initNewConv2d()
            self.__addFilterConv2d(self.conv2, oldConv2d_1)
            #self.__addFilterConv2d(self.conv3, oldConv2d_2)

            self.conv3.weight = torch.nn.Parameter(oldConv2d_2.weight.clone())
            self.conv3.bias = torch.nn.Parameter(oldConv2d_2.bias.clone())

            self.fc1.weight = torch.nn.Parameter(oldLinear.weight.clone())
            self.fc1.bias = torch.nn.Parameter(oldLinear.bias.clone())
        else:
            print("wrong type network")


    def __addFilterConv2d(self, newConv2d, oldConv2d):

        oldFilter = oldConv2d.weight.clone()
        oldBias = oldConv2d.bias.clone()

        shape = oldConv2d.weight.shape
        
        resized = torch.zeros(shape[0], 1, shape[2], shape[3]).cuda()

        oldFilter = torch.cat((oldFilter, resized), dim=1)


        for i in range(shape[0]):
            oldFilter[i][shape[1]] = oldFilter[i][shape[1]-1].clone()

        newConv2d.weight = torch.nn.Parameter(oldFilter)
        newConv2d.bias = torch.nn.Parameter(oldBias)

        del resized

    def __initNewConv2d(self):
        factor_n = 0.25
        entries = 3

        torch.nn.init.constant_(self.conv1.weight, factor_n / entries)
        torch.nn.init.constant_(self.conv1.bias, 0)

    def generateValue(self, inputValue, outputValue):

        value = None
        inputShape = inputValue.shape
        outputShape = outputValue.shape

        diff_kernel = abs(inputShape[2] - outputShape[2])
        
        if inputShape[2] >= outputShape[2]:
            
            newValue = outputValue.data.clone()
            newValue = torch.nn.functional.pad(newValue,(0, diff_kernel, 0, diff_kernel),"constant", 0)

            value = torch.cat((inputValue, newValue), dim=1)
        else:
            print("OUTPUT MAYOR A INPUT")
        
        return value

    def propagation_1(self, x):

        shapeFilter = self.conv1.weight.shape
        normalize = shapeFilter[2] * shapeFilter[3]
        input_conv1 = x

        x = self.conv1(x) / normalize

        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)

        x = self.generateValue(input_conv1, x)

        shapeFilter = self.conv2.weight.shape
        normalize = shapeFilter[2] * shapeFilter[3]

        x = self.conv2(x) / normalize
        
        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)
    
        shape = x.shape
        x = x.view(shape[0], -1)

        x = self.fc1(x)

        return x

    def propagation_2(self, x):

        ## conv1
        shapeFilter = self.conv1.weight.shape
        normalize = shapeFilter[2] * shapeFilter[3]
        input_conv1 = x

        x = self.conv1(x) / normalize

        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)

        x = self.generateValue(input_conv1, x)

        ## conv2
        shapeFilter = self.conv2.weight.shape
        normalize = shapeFilter[2] * shapeFilter[3]

        x = self.conv2(x) / normalize
        
        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)
        
        x = self.generateValue(input_conv1, x)
        
        ## conv3
        shapeFilter = self.conv3.weight.shape
        normalize = shapeFilter[2] * shapeFilter[3]

        x = self.conv3(x) / normalize
        
        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)

        ## Linear
        shape = x.shape
        x = x.view(shape[0], -1)

        x = self.fc1(x)

        return x

    def forward(self, x):
        
        loss = None

        if self.typeNet == 1:
            loss = self.propagation_1(x)
        else:
            loss = self.propagation_2(x)
        
        return loss
    

def Test_pytorchNetwork(batch, x, y, n_images):

    print("###### TEST #1 ######")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net_1 = NetPytorch(typeNet=1).to(device)
    net_2 = NetPytorch(typeNet=2).to(device)

    net_1.upgradeFlag(False)
    net_2.upgradeFlag(False)
    net_2.recieveParameters(oldConv2d_1=net_1.conv1, oldConv2d_2=net_1.conv2, oldLinear=net_1.fc1)
    net_1.upgradeFlag(True)
    net_2.upgradeFlag(True)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.CrossEntropyLoss()

    optimizer_1 = optim.SGD(net_1.parameters(), lr=0.01, momentum=0)
    optimizer_2 = optim.SGD(net_2.parameters(), lr=0.01, momentum=0)

    energy_1 = 0
    energy_2 = 1
    stop = False
    i = 0

    for _,a in enumerate(batch):
        
        inputs, labels = a[0], a[1]

        while stop == False:
            
            energy_1 = 0.0
            optimizer_1.zero_grad()
            outputs = net_1(inputs)
            loss_1 = criterion_1(outputs, labels)
            loss_1.backward()
            optimizer_1.step()
            energy_1 += loss_1.item()

            energy_2 = 0.0
            optimizer_2.zero_grad()
            outputs = net_2(inputs)
            loss_2 = criterion_2(outputs, labels)
            loss_2.backward()
            optimizer_2.step()
            energy_2 += loss_2.item()

            i += 1

            if energy_2 < energy_1:
                stop = True
            
            if i % 100 == 99:   
                print("Energy #1: ", energy_1, "i = ", i)
                print("Energy #2: ", energy_2, "i = ", i)
    
    print("last Iteration=", i)
    print("energy network 1=", energy_1)
    print("energy network 2=", energy_2)

def Test_pytorch_energy(batch, x, y, n_images):

    print("###### TEST #2 ######")
    
    networkADN = ((0, 3, 3, x-1, y-1), (0, 6, 3, x, y), (1, 3, 2), (2,))
    mutationADN = ((0, 3, 3, x-1, y-1), (0, 6, 3, x-1, y-1) , (0, 9, 3, x, y), (1, 3, 2), (2,))
    #mutationADN = ((0, 3, 4, x, y), (1, 4, 2), (2,))
    network = nw_p.Network(networkADN, cudaFlag=True)
    clone = MutateNetwork.executeMutation(network, mutationADN)

    energy_2_filters = 0
    energy_3_filters = 1
    stop = False
    i = 0
    for _,a in enumerate(batch):

        
        while stop == False:
            #print("entrenando red #1")
            network.Training(data=a[0], p=1, dt=0.01, labels=a[1])
            #print("entrenando red #2")
            clone.Training(data=a[0], p=1, dt=0.01, labels=a[1])
            energy_2_filters = network.total_value
            energy_3_filters = clone.total_value
            i += 1

            if energy_3_filters < energy_2_filters:
                stop = True

            if i % 100 == 99:
                print("energy network 1=", energy_2_filters)
                print("energy network 2=", energy_3_filters)
        
    
    print("last Iteration=", i)
    print("energy network 1=", energy_2_filters)
    print("energy network 2=", energy_3_filters)



dataGen = GeneratorFromImage.GeneratorFromImage(2, 160, cuda=True)
dataGen.dataConv2d()
size = dataGen.size

x = size[1]
y = size[2]
k = 2

batch = [dataGen.data]

Test_pytorchNetwork(batch, x, y, len(dataGen.data[0]))
Test_pytorch_energy(batch, x, y, len(dataGen.data[0]))