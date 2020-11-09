import torch
import torch.nn as nn
import torch.tensor as tensor


class MemoryUnit():

    def __init__(self, kernel_size, in_channels, in_channels_candidate, out_channels, cuda_flag=True):

        self.cuda_flag = cuda_flag
        self.__createStructure(kernel_size=kernel_size,in_channels=in_channels, out_channels=out_channels, in_channels_candidate= in_channels_candidate)
        self.ct = None
        self.ht = None
        self.xt = None

    
    def __createStructure(self, kernel_size, in_channels, out_channels, in_channels_candidate):

        self.convFt = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.convIt = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.convCand = nn.Conv1d(in_channels_candidate, out_channels, kernel_size)
        self.convOt = nn.Conv1d(in_channels, out_channels, kernel_size)

        if self.cuda_flag == True:
            self.convFt.cuda()
            self.convIt.cuda()
            self.convCand.cuda()
            self.convOt.cuda()

    def compute(self, xt, last_ht=None, last_ct=None):

        current_input_ct = None
        current_input_ht = None

        clone_ht = None
        clone_ct = None

        self.xt = xt

        if last_ht is not None:
            
            clone_ct = last_ct.clone()
            clone_ct.transpose_(1,2)            
            clone_ct = torch.reshape(clone_ct, (-1,xt.shape[1],xt.shape[2]))
            current_input_ht = torch.cat((clone_ct, last_ht, xt), dim=1)
            current_input_candidates = torch.cat((last_ht, xt), dim=1)
        else:
            current_input_ht = xt
            current_input_candidates = xt
        
        self.__computeCt(current_input_ht, current_input_candidates, last_ct)

        if last_ht is not None:
            
            clone_ct = self.ct.clone()
            clone_ct.transpose_(1,2)            
            clone_ct = torch.reshape(clone_ct, (-1,xt.shape[1],xt.shape[2]))

            current_input_ct = torch.cat((clone_ct, last_ht, xt), dim=1)
        else:
            current_input_ct = xt

        self.__computeHt(current_input_ct)

    
    def __computeCt(self, current_input, current_input_candidates, last_ct=None):

        sigmoid_ft = torch.nn.Sigmoid()
        sigmoid_it = torch.nn.Sigmoid()
        tanh_cand = torch.nn.Tanh()

        ft = self.convFt(current_input)
        ft = sigmoid_ft(ft)

        it = self.convIt(current_input)

        it = sigmoid_it(it)

        candidates = self.convCand(current_input_candidates)
        candidates = tanh_cand(candidates)
        
        if last_ct is not None:
            ft = ft * last_ct

        a = it * candidates

        self.ct = ft + a


    def __computeHt(self, current_input):
        
        sigmoid_ot = torch.nn.Sigmoid()
        tanh_ct = torch.nn.Tanh()

        ot = self.convOt(current_input)

        ot = sigmoid_ot(ot)
        
        a = tanh_ct(self.ct)

        self.ht = ot * a

        self.ht.transpose_(1, 2)
        self.ht = torch.reshape(self.ht, (-1, self.xt.shape[1], self.xt.shape[2]))


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


        
        

