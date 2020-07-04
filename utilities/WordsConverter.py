import Factory.TensorFactory as TensorFactory

class WordsConverter():

    def __init__(self, cuda):
        self.cuda = cuda
        self.__createDictionary()

    def __createDictionary(self):
        self.letter_dictionary = {}

        self.letter_dictionary["a"] = 0
        self.letter_dictionary["b"] = 1
        self.letter_dictionary["c"] = 2
        self.letter_dictionary["d"] = 3
        self.letter_dictionary["e"] = 4
        self.letter_dictionary["f"] = 5
        self.letter_dictionary["g"] = 6
        self.letter_dictionary["h"] = 7
        self.letter_dictionary["i"] = 8
        self.letter_dictionary["j"] = 9
        self.letter_dictionary["k"] = 10
        self.letter_dictionary["l"] = 11
        self.letter_dictionary["m"] = 12
        self.letter_dictionary["n"] = 13
        self.letter_dictionary["o"] = 14
        self.letter_dictionary["p"] = 15
        self.letter_dictionary["q"] = 16
        self.letter_dictionary["r"] = 17
        self.letter_dictionary["s"] = 18
        self.letter_dictionary["t"] = 19
        self.letter_dictionary["u"] = 20
        self.letter_dictionary["v"] = 21
        self.letter_dictionary["w"] = 22
        self.letter_dictionary["x"] = 23
        self.letter_dictionary["y"] = 24
        self.letter_dictionary["z"] = 25
        self.letter_dictionary[" "] = 26
        self.letter_dictionary["#"] = 27

        self.array = []
        for i in range(len(self.letter_dictionary)):
            self.array.append(0)
    
    def __getLongestWordValue(self, wordsArray):
        
        letters = 0

        for word in wordsArray:
            lenght = len(word)

            if lenght >= letters:
                letters = lenght
        
        return letters
    
    def convertWordsToTensor(self, wordsArray):
        words_amount = len(wordsArray)
        max_letters = self.__getLongestWordValue(wordsArray)
        letter_size = len(self.array)

        tensorValue = TensorFactory.createTensorZeros(tupleShape=(words_amount, max_letters, letter_size), cuda=self.cuda)
        i = 0
        for word in wordsArray:
            j = 0
            for letter in word:
                tensorValue[i][j] = self.getTensor(letter)
                j += 1
            i += 1

        return tensorValue


    def getTensor(self, letter):

        value = None
        index = self.letter_dictionary.get(letter)

        
        if index is not None:
            value = self.__createLetter(index)
        else:
            print("letter doesnt exist=", letter)

        return value

    def __createLetter(self, index):

        self.array[index] = 1
        value = TensorFactory.createTensorValues(self.array, cuda=self.cuda)
        self.array[index] = 0

        return value
    
    def indexToLetter(self, index):
        
        value = None
        for key in self.letter_dictionary.keys():
            compare = self.letter_dictionary.get(key)
            if self.letter_dictionary.get(key) == index:
                value = key
        
        return value