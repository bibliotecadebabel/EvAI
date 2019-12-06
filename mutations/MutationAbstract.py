from abc import ABC, abstractmethod

class Mutation(ABC):

    @abstractmethod
    def doMutate(self, network, newAdn):
        pass

