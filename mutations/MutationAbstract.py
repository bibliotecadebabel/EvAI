from abc import ABC, abstractmethod

class Mutation(ABC):

    @abstractmethod
    def doMutate(self, oldFilter, oldBias, newNode):
        pass

