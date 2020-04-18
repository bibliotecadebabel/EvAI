import torch
import torch.nn as nn
import torch.tensor as tensor


class InternalModule():

    def __init__(self, kernelSize, inChannels, outChannels, cudaFlag=True):

        self.cudaFlag = cudaFlag
        self.kernelSize = kernelSize
        self.__createStructure(inChannels=inChannels, outChannels=outChannels)
        self.ct = None
        self.ht = None

    
    def __createStructure(self, inChannels, outChannels):
        self.convFt = nn.Conv1d(inChannels, outChannels, self.kernelSize)
        self.convIt = nn.Conv1d(inChannels, outChannels, self.kernelSize)
        self.convCand = nn.Conv1d(inChannels, outChannels, self.kernelSize)
        self.convOt = nn.Conv1d(inChannels, outChannels, self.kernelSize)

        if self.cudaFlag == True:
            self.convFt.cuda()
            self.convIt.cuda()
            self.convCand.cuda()
            self.convOt.cuda()

    def compute(self, xt, last_ht=None, last_ct=None):
        
        currentInput = None
        if last_ht is not None:
            currentInput = torch.cat((last_ht, xt), dim=1)
        else:
            currentInput = xt
        
        self.__computeCt(currentInput, last_ct)
        self.__computeHt(currentInput)

    
    def __computeCt(self, currentInput, last_ct=None):

        sigmoid_ft = torch.nn.Sigmoid()
        sigmoid_it = torch.nn.Sigmoid()
        tanh_cand = torch.nn.Tanh()

        ft = self.convFt(currentInput)
        ft = sigmoid_ft(ft)

        it = self.convIt(currentInput)
        it = sigmoid_it(it)

        candidates = self.convCand(currentInput)
        candidates = tanh_cand(candidates)
        
        if last_ct is not None:
            ft = ft * last_ct

        a = it * candidates

        self.ct = ft + a

    def __computeHt(self, currentInput):

        sigmoid_ot = torch.nn.Sigmoid()
        tanh_ct = torch.nn.Tanh()

        ot = self.convOt(currentInput)
        ot = sigmoid_ot(ot)
        
        a = tanh_ct(self.ct)

        self.ht = ot * a

        self.ht.transpose_(1, 2)


        
        

