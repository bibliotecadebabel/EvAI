from Node_class import Node
import numpy as np
import Interfaces as Inter
import Net as Net0
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.modules.pooling as P

N = 20
M = 4
A = Variable(torch.randn(1,1,N,N ))
M = Variable(torch.ones(1,1, M, M))
#output = F.conv2d(A, M)
#print(output)
a=np.load('w.npy')
data=np.load('data.npy')
b=torch.from_numpy(a[0]).float()
b.unsqueeze_(0)
print('Convertion completed')
print('hi')
c=torch.from_numpy(Inter.Image2array('screena')).float()
c.unsqueeze_(0)
print('Convertion completed')
f=F.conv2d(A, M)
print('Convolution completed')
print(f.size())
print(b.size())
print(c.size())
print(M.size())
print(A.size())
print(M)
print('Afet pooking we get')
m = P.MaxPool2d(2, stride=2)
pool = P.MaxPool1d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool1d(2, stride=2)
input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8, 9]]]
output, indices = pool(input)


#print(F.MaxPool2d(A))
"""a=np.load('w.npy')
Inter.Array2image(a[0],'w0')
size=np.shape(a)
Net=Net0.Network((size[0],size[1]))
Net.w_node.Value=a
data=np.load('data.npy')
print(Net.pre(data[0]))
print(Net.pre(data[1000]))
Inter.traking(Net,'screen','map')"""
