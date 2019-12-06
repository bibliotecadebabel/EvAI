from mutations.MutationAbstract import Mutation

class AddFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, network, newAdn):
        print("Mutate AddFilter Conv2d")
        
class RemoveFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, network, newAdn):
        print("Mutate RemoveFilter Conv2d")