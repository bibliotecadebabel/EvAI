import LSTM.InternalModule as IM
import torch
import torch.nn as nn
import torch.tensor as tensor



batch = 1
words = 10
word_size = 6
output_size = 6
CUDA = False

xt = torch.rand(batch, words, word_size)

if CUDA == True:
    xt = xt.cuda()


module = IM.InternalModule(kernelSize=word_size, inChannels=words, outChannels=output_size, cudaFlag=CUDA)
module.compute(xt)

print("ct size=", module.ct.size())
print("ht size=", module.ht.size())

