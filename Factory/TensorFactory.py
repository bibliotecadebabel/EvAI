import torch

def createTensor(body, cuda=True, requiresGrad=False):

    value = torch.tensor(body, dtype=torch.long, requires_grad=requiresGrad)

    if cuda == True:
        value.cuda()

    return value

def createTensorRand(tupleShape, cuda=True, requiresGrad=False):

    value = torch.rand((tupleShape), dtype=torch.float32, requires_grad=requiresGrad)

    if cuda == True:
        value.cuda()

    return value