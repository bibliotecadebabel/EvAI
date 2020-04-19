import torch

def createTensor(body, cuda=True, requiresGrad=False):

    if cuda == True:
        value = torch.tensor(body, dtype=torch.long, requires_grad=requiresGrad).cuda()
    else:
        value = torch.tensor(body, dtype=torch.long, requires_grad=requiresGrad)

    return value

def createTensorRand(tupleShape, cuda=True, requiresGrad=False):

    value = torch.rand((tupleShape), dtype=torch.float32, requires_grad=requiresGrad)

    if cuda == True:
        value.cuda()

    return value

def createTensorZeros(tupleShape, cuda=True, requiresGrad=False):

    value = torch.zeros((tupleShape), dtype=torch.float32, requires_grad=requiresGrad)

    if cuda == True:
        value = value.cuda()

    return value

def createTensorValues(values, cuda=True, requiresGrad=False):

    value = torch.tensor(values, dtype=torch.float32, requires_grad=requiresGrad)

    if cuda == True:
        value = value.cuda()
    
    return value

def createTensorOnes(tupleShape, cuda=True, requiresGrad=False):
    value = torch.ones((tupleShape), dtype=torch.float32, requires_grad=requiresGrad)

    if cuda == True:
        value = value.cuda()

    return value