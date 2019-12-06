from mutations.MutationAbstract import Mutation

class AddFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, network, newAdn):
        print("Mutate AddFilter Linear")
        
class RemoveFilterMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def doMutate(self, network, newAdn):
        print("Mutate RemoveFilter Linear")