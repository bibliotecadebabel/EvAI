from Node_class import Node
import numpy as np
import Interfaces as Inter
import Net as Net0
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.modules.pooling as P

N = 200
M = 10
A = Variable(torch.randn(1,3,N,N ))
M = Variable(torch.ones(1,3, M, M))
#output = F.conv2d(A, M)
#print(output)
f=F.conv2d(A, M)
