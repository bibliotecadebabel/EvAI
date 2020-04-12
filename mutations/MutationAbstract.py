#from abc import ABC, abstractmethod, abc
import abc

class Mutation(abc.ABC):
    
    def value_getter(self):
        return 0
    
    def value_setter(self, newvalue):
        return

    value = abc.abstractproperty(value_getter, value_setter)

    @abc.abstractmethod
    def doMutate(self, oldFilter, oldBias, newNode):
        pass

