import LSTM.NetworkLSTM as netLSTM
import torch
import torch.nn as nn
import torch.tensor as tensor
import Factory.WordsConverter as WordsConverter

CUDA = False

wordsConverter = WordsConverter.WordsConverter(cuda=CUDA)

words = ["hola ", "como ", "paralelepipedo ", "estas"]

words_tensor = wordsConverter.convertWordsToTensor(words)

word_amount = words_tensor.shape[0]
letters_max = words_tensor.shape[1]
kernel_size = words_tensor.shape[2]


network = netLSTM.NetworkLSTM(max_letters=letters_max, inChannels=1, outChannels=kernel_size, kernelSize=kernel_size)

network.Training(data=words_tensor, dt=0.01, p=1)




