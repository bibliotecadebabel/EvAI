import LSTM.NetworkLSTM as netLSTM
import torch
import torch.nn as nn
import torch.tensor as tensor
import utilities.WordsConverter as WordsConverter

CUDA = True
words = [" hi", " oj", " lk", " ab"]

modulo_0 ="    "
modulo_1 = "hola"
module_2 = "ijkb"
wordsConverter = WordsConverter.WordsConverter(cuda=CUDA)
'''
current_path ="  "
alazar: h,o,l,a
words = [" hh", " oo", " ll", " aa"]
current_path = "  l"
rpedict: " l"
produce: 
al azar: j k
words_2 = [" ll", " lh", " lk", " lj"]
current_path = " lk"
predict: "lk"
produce: k j
alzar: m n
words_3 = ["lkk", "lkj", "lkm", "lkn"]
current_path "lkn"
predict = "kn"
produce: k,j
'''
words_tensor = wordsConverter.convertWordsToTensor(words)
print("input tensor: ", words_tensor.size())

print(words_tensor)

word_amount = words_tensor.shape[0]
letters_max = words_tensor.shape[1]
kernel_size = words_tensor.shape[2]


network = netLSTM.NetworkLSTM(max_letters=letters_max, inChannels=1, outChannels=kernel_size, kernelSize=kernel_size)

print("empezando entrenamiento")
network.Training(data=words_tensor, dt=0.001, p=100)
print("entrenamiento finalizado")

predict_value = input("introduce el inicio para predecir: ")



input_predict = len(predict_value)

stop = False
while stop == False:

    value = [predict_value]
    words_predict = wordsConverter.convertWordsToTensor(value)
    
    index = network.predict(words_predict)

    if index != 27:
        next_letter = wordsConverter.indexToLetter(index)
        predict_value = predict_value + next_letter

        print("Prediccion: ", predict_value)
    else:
        stop = True








