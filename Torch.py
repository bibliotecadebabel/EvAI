import torch
from DAO import GeneratorFromCIFAR
import torch.nn as nn
import torch.nn.functional as F
import time
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3).cuda()
        self.conv2 = nn.Conv2d(32, 32, 3).cuda()
        self.pool1 = nn.MaxPool2d(2, stride=None).cuda()
        self.conv3 = nn.Conv2d(32, 64, 3).cuda()
        self.conv4 = nn.Conv2d(64, 64, 3).cuda()
        self.pool2 = nn.MaxPool2d(2, stride=None).cuda()
        self.fc1 = nn.Linear(64 * 5 * 5, 128).cuda()
        self.fc2 = nn.Linear(128, 10).cuda()    
    def forward(self, x):

        dropout = torch.nn.Dropout2d(p=0.10)
        x = dropout(x)
        x = F.relu(self.conv1(x))

        dropout = torch.nn.Dropout2d(p=0.10)
        x = dropout(x)
        x = F.relu(self.conv2(x))

        x = self.pool1(x)

        dropout = torch.nn.Dropout2d(p=0.10)
        x = dropout(x)
        x = F.relu(self.conv3(x))

        dropout = torch.nn.Dropout2d(p=0.10)
        x = dropout(x)
        x = F.relu(self.conv4(x))
        
        x = self.pool2(x)

        dropout = torch.nn.Dropout2d(p=0.10)
        x = dropout(x)

        x = x.view(-1, 64 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x



def test():

    dataGen = GeneratorFromCIFAR.GeneratorFromCIFAR(2,  64, threads=2)
    dataGen.dataConv2d()
    net = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(50):  # loop over the dataset multiple times
        
        running_loss = 0.0
        start_time = time.time()

        for i, data in enumerate(dataGen._trainoader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].cuda(), data[1].cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        end_time = time.time()
        print("epoch time: ", end_time - start_time)

    correct = 0
    total = 0
    net = net.eval()
    with torch.no_grad():
        for i, data in enumerate(dataGen._testloader):
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

if __name__ == "__main__":
    test()