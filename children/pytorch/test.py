import Network as Net
import torch

import torchvision
import torchvision.transforms as transforms

x = 10
y = 10
k = 10

'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
objects = [x,y,k]
network = Net.Network(objects)

Net.functions.Propagation(network.nodes[3].objects[0])
#print("value D: ", network.nodes[3].objects[0].value)
network.nodes[3].objects[0].value.backward()

network.print_params()

print("\n \n")
network.nodes[3].objects[0].getBias()
network.nodes[3].objects[0].getFilter()
#print(network.nodes[0].objects[0].object.grad.data)
#print(list(network.nodes[0].objects[0].object.parameters()))