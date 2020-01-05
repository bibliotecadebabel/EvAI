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

    def setAttribute(self, name, value):
        setattr(self,name,value)
    
    def __getAttribute(self, name):

        attribute = None
        try:
            attribute = getattr(self, name)
        except AttributeError:
            pass

        return attribute
    
    def deleteAttribute(self, name):
        try:
            delattr(self, name)
        except AttributeError:
            pass
    
    def executeLayer(self, layerName, x):

        layer = self.__getAttribute(layerName)

        if layer is not None:
            
            return layer(x)
        
        else:

            return x

    def forward(self, x):

        x = self.executeLayer("conv1", x)

        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)

        x = self.executeLayer("conv2", x)
        
        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)
        
        x = self.executeLayer("conv3", x)

        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)

        x = x.view(-1, 12 * 1 * 1)

        x = self.executeLayer("fc1", x)
        x = self.executeLayer("fc2", x)

        return x

def Test_dynamicNetwork(dataGen):
    batch = [dataGen.data]
    k = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net([dataGen.size[1], dataGen.size[2], k]).to(device)

    net.setAttribute("conv1", nn.Conv2d(3, 50, 2, 2).cuda())
    net.setAttribute("conv2", nn.Conv2d(50, 25, 2, 2).cuda())
    net.setAttribute("conv3", nn.Conv2d(25, 12, 2, 2).cuda())
    net.setAttribute("fc1", nn.Linear(12 * 1 * 1, 10).cuda())
    net.setAttribute("fc2", nn.Linear(10 * 1 * 1, 2).cuda())

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
                #net.deleteAttribute("fc2")
                running_loss = 0.0


dataGen = GeneratorFromImage.GeneratorFromImage(2, 100, cuda=True)
dataGen.dataConv2d()
size = dataGen.size


x = size[1]
y = size[2]
k = 2

Test_dynamicNetwork(dataGen)