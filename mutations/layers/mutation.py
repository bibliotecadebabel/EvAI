import abc

class Mutation(abc.ABC):
    
    def value_getter(self):
        return 0
    
    def value_setter(self, newvalue):
        return

    value = abc.abstractproperty(value_getter, value_setter)

    @abc.abstractmethod
    def execute(self, oldFilter, oldBias, newNode):
        pass

