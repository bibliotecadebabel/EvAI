import torch

def deleteCache(object_pytorch):

    del object_pytorch
    torch.cuda.empty_cache()
