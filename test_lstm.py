import LSTM.NetworkLSTM as netLSTM
import torch
import torch.nn as nn
import torch.tensor as tensor
import Factory.WordsConverter as WordsConverter

CUDA = True

wordsConverter = WordsConverter.WordsConverter(cuda=CUDA)

"hola felix morales hola desire mendoza"
words = ["hola felix morales", "felix morales hola", "morales hola desire", "hola desire mendoza", "desire mendoza", "mendoza"]
#print(words)

words_tensor = wordsConverter.convertWordsToTensor(words)

print(words_tensor.size())

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








