import children.pytorch.Network as nw
import children.Interfaces as Inter
from DAO import GeneratorFromImage

import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, objects):
        super(Net, self).__init__()
        self.objects = objects

    def set_attribute(self, name, value):
        setattr(self,name,value)
    
    def __get_attribute(self, name):

        attribute = None
        try:
            attribute = getattr(self, name)
        except AttributeError:
            pass

        return attribute
    
    def delete_attribute(self, name):
        try:
            delattr(self, name)
        except AttributeError:
            pass
    
    def executeLayer(self, layerName, x):

        layer = self.__get_attribute(layerName)

        if layer is not None:
            
            return layer(x)
        
        else:

            return x

    def forward(self, x):

        x = self.executeLayer("conv1", x)

        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(x) + torch.nn.functional.relu(x)

        x = x.view(-1, 100 * 1 * 1)
        x = self.executeLayer("fc1", x)

        return x

def Test_dynamicNetwork(dataGen):
    batch = [dataGen.data]
    k = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    objects = [dataGen.size[1], dataGen.size[2],k]
    net = Net(objects).to(device)

    net.set_attribute("conv1", nn.Conv2d(3, objects[2], objects[0], objects[1]).cuda())
    net.set_attribute("fc1", nn.Linear(objects[2] * 1 * 1, 2).cuda())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.1)
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
                #net.delete_attribute("fc2")
                running_loss = 0.0


dataGen = GeneratorFromImage.GeneratorFromImage(2, 15000, cuda=True)
dataGen.dataConv2d()
size = dataGen.size


x = size[1]
y = size[2]
k = 2

Test_dynamicNetwork(dataGen)