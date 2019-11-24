import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.modules.pooling as P

N = 20
M = 4
A = Variable(torch.randn(1,1,N,N))
M = Variable(torch.ones(1,1, M, M))
#print(A)
#print(M)
output = F.conv2d(A, M)
print(output.size())
Matrix=nn.Linear(10,5)
x=torch.ones(10)
print(x.size())
print('The result of the multiplication is')
print(F.relu(Matrix(x)))
print('The parameters of Matrix are')
k=0
for f in Matrix.parameters():
    k=k+1
    print('The parameter number')
    print(k)
    print('is:')
    print(f)
para=list(Matrix.parameters())
print('The size of the first parameter is')
print(para[0])
para[0]=para[0]*0
print('The new value is')
print(para[0])
para=list(Matrix.parameters())
print('The new value is')
print(para[0])
print('Done')
Matrix.weight.data=Matrix.weight.data*0
Matrix.bias.data=Matrix.bias.data*0
print((Matrix.weight.data))
print('The result of the multiplication is')
print(Matrix(x))
#a=np.load('w.npy')
#data=np.load('data.npy')
#b=torch.from_numpy(a[0]).float()
#b.unsqueeze_(0)
#print('Convertion completed')
#print('hi')
#c=torch.from_numpy(Inter.Image2array('screena')).float()
#c.unsqueeze_(0)
#print('Convertion completed')
#f=F.conv2d(A, M)
#print('Convolution completed')
#print(f.size())
#print(b.size())
#print(c.size())
#print(M.size())
#print(A.size())
#print(M)
