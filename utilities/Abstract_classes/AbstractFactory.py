from abc import ABC, abstractmethod

class FactoryClass(ABC):

    def __init__(self):

        self.dictionary = {}
        self.createDictionary()

    @abstractmethod
    def createDictionary(self):
        pass

    def findValue(self, key):

        return self.dictionary[key]
