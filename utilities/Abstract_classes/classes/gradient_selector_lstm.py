from utilities.Abstract_classes.AbstractSelector import Selector, Observation

class Gradient_selector_lstm(Selector):
    class Observation_creator(Observation):
        def __init__(self):
            super().__init__()
    def __init__(self):
        super().__init__(self.Observation_creator)

    def Action2tensor(self,action):
        pass

    def create_net(self):
        pass

    def register_observations(self, space):
        pass

    def space2action(self,space):
        pass

    def forget_filters(self,observation):
        pass

    def train(self):
        pass

    def update_predicted_actions(self):
        pass

    def get_predicted_actions(self):
        pass




    """def charge_node(self,key,charger=None):
        node=self.Graph.key2node[key]
        node.objects.append(0)"""
