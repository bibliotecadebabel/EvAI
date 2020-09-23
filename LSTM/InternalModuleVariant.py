import torch
import torch.nn as nn
import torch.tensor as tensor


class InternalModuleVariant():

    def __init__(self, kernelSize, inChannels, outChannels, cuda_flag=True):

        self.cuda_flag = cuda_flag
        self.kernelSize = kernelSize
        self.__createStructure(inChannels=inChannels, outChannels=outChannels)
        self.ct = None
        self.ht = None
        self.xt = None

    
    def __createStructure(self, inChannels, outChannels):
        self.convFt = nn.Conv1d(inChannels, outChannels, self.kernelSize)
        self.convIt = nn.Conv1d(inChannels, outChannels, self.kernelSize)
        self.convCand = nn.Conv1d(inChannels, outChannels, self.kernelSize)
        self.convOt = nn.Conv1d(inChannels, outChannels, self.kernelSize)

        if self.cuda_flag == True:
            self.convFt.cuda()
            self.convIt.cuda()
            self.convCand.cuda()
            self.convOt.cuda()

    def compute(self, xt, last_ht=None, last_ct=None):

        currentInput_ct = None
        currentInput_ht = None

        clone_ht = None
        clone_ct = None

        self.xt = xt

        if last_ht is not None:
            
            clone_ct = last_ct.clone()
            clone_ct.transpose_(1,2)            
            clone_ct = torch.reshape(clone_ct, (-1,xt.shape[1],xt.shape[2]))
            currentInput_ht = torch.cat((clone_ct, last_ht, xt), dim=1)
        else:
            currentInput_ht = xt
        
        self.__computeCt(currentInput_ht, last_ct)

        if last_ht is not None:
            
            clone_ct = self.ct.clone()
            clone_ct.transpose_(1,2)            
            clone_ct = torch.reshape(clone_ct, (-1,xt.shape[1],xt.shape[2]))

            currentInput_ct = torch.cat((clone_ct, last_ht, xt), dim=1)
        else:
            currentInput_ct = xt

        self.__computeHt(currentInput_ct)

    
    def __computeCt(self, currentInput, last_ct=None):

        sigmoid_ft = torch.nn.Sigmoid()
        sigmoid_it = torch.nn.Sigmoid()
        tanh_cand = torch.nn.Tanh()

        #print("current input ct: ",currentInput.size())
        ft = self.convFt(currentInput)
        #print("ft: ", ft.size())
        ft = sigmoid_ft(ft)

        it = self.convIt(currentInput)
        #print("it: ", it.size())
        it = sigmoid_it(it)

        candidates = self.convCand(currentInput)
        candidates = tanh_cand(candidates)
        
        if last_ct is not None:
            ft = ft * last_ct

        a = it * candidates

        self.ct = ft + a
        #print("ct: ", self.ct.size())

    def __computeHt(self, currentInput):
        
        sigmoid_ot = torch.nn.Sigmoid()
        tanh_ct = torch.nn.Tanh()

        ot = self.convOt(currentInput)
        #print("output ht: ", ot.size())
        ot = sigmoid_ot(ot)
        
        a = tanh_ct(self.ct)

        #print("a: ", a.shape)
        #print("ot: ", ot.shape)
        self.ht = ot * a
        #print("output ht: ", self.ht.size())
        self.ht.transpose_(1, 2)
        self.ht = torch.reshape(self.ht, (-1, self.xt.shape[1], self.xt.shape[2]))
        #print("ht: ", self.ht.size())

    def set_grad_flag(self, flag):

        self.convFt.weight.requires_grad = flag
        self.convFt.bias.requires_grad = flag

        if self.convFt.weight.grad is not None:
            self.convFt.bias.grad.requires_grad = flag
            self.convFt.weight.grad.requires_grad = flag

        self.convIt.weight.requires_grad = flag
        self.convIt.bias.requires_grad = flag

        if self.convIt.weight.grad is not None:
            self.convIt.bias.grad.requires_grad = flag
            self.convIt.weight.grad.requires_grad = flag

        self.convCand.weight.requires_grad = flag
        self.convCand.bias.requires_grad = flag

        if self.convCand.weight.grad is not None:
            self.convCand.bias.grad.requires_grad = flag
            self.convCand.weight.grad.requires_grad = flag

        self.convOt.weight.requires_grad = flag
        self.convOt.bias.requires_grad = flag

        if self.convOt.weight.grad is not None:
            self.convOt.bias.grad.requires_grad = flag
            self.convOt.weight.grad.requires_grad = flag


        
        

