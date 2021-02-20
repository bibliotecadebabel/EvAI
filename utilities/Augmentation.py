import numpy as np
import torch

ENABLE_EXTRA = False

class Ricap():

    def __init__(self, beta):
        self.__beta = beta
        self.__w = {}
        self.__c = {}

    # Autor = 4ui_iurz1 (2019)
    # Repositorio de Github = https://github.com/4uiiurz1/pytorch-ricap    
    def doRicap(self, inputs, target, cuda=True):

        I_x, I_y = inputs.size()[2:]

        w = int(np.round(I_x * np.random.beta(self.__beta, self.__beta)))
        h = int(np.round(I_y * np.random.beta(self.__beta, self.__beta)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]

        cropped_images = {}
        c_ = {}
        W_ = {}

        for k in range(4):
            idx = torch.randperm(inputs.size(0))
            x_k = np.random.randint(0, I_x - w_[k] + 1)
            y_k = np.random.randint(0, I_y - h_[k] + 1)
            cropped_images[k] = inputs[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
            
            if cuda == True:
                c_[k] = target[idx].cuda()
            else:
                c_[k] = target[idx]
                
            W_[k] = w_[k] * h_[k] / (I_x * I_y)

        self.__c = c_
        self.__w = W_

        patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
            torch.cat((cropped_images[2], cropped_images[3]), 2)),
        3)

        return patched_images

    # Autor = 4ui_iurz1 (2019)
    # Repositorio de Github = https://github.com/4uiiurz1/pytorch-ricap   
    def generateLoss(self, layer):

        parent_layer = layer.node.parents[0].objects[0]
        output = parent_layer.value

        loss = sum([self.__w[k] * layer.object(output, self.__c[k]) for k in range(4)])

        return loss