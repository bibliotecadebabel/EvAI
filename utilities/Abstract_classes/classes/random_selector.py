from utilities.Abstract_classes.AbstractSelector import Selector, Observation
import random

class random_selector(Selector):
    class Observation_creator(Observation):
        def __init__(self,num_layer=1):
            super().__init__()
            self.num_layer=num_layer
    def __init__(self,num_actions=5,directions=(
        (0,1,0,0),
        (1,0,0,0),
        (0,0,1,1),
        (0,0,1)
        )):
        super().__init__(self.Observation_creator)
        self.directions=directions
        self.num_actions=num_actions
        self.max_observation_size = 1

    def Action2tensor(self,action):
        pass

    def create_net(self):
        pass

    def register_observations(self, space):
        if type(space) is tuple:
            num_conv_layers = len([0 for layer in space if layer[0] == 0])
            self.observations.append(self.observation_creator(num_conv_layers))
        elif space.node_max_particle:
            DNA=space.graph.nodeek2key
            self.register_observations(self, DNA)
        else:
            self.register_observations(self, space.center)
        pass

    def print_observation(self):
        if self.observations:
            observation=self.observations[0]
            print(observation.num_layer)
        else:
            print(None)

    def space2action(self,space):
        return None

    def forget_weight(self,observation):
        pass

    def train(self):
        pass

    def update_predicted_actions(self):
        if self.observations:
            observation=self.observations[0]
            num_layer=observation.num_layer
            num_directions=len(self.directions)
            k=0
            while k<self.num_actions:
                layer=random.randint(0,num_layer)
                direction=random.randint(0,num_directions-1)
                if not ([layer,direction] in self.predicted_actions):
                    self.predicted_actions.append([layer,direction])
                k=k+1
        else:
            self.predicted_actions=None


    def get_predicted_actions(self):
        return tuple([(action[0],self.directions[action[1]])
        for action in self.predicted_actions])




    """def charge_node(self,key,charger=None):
        node=self.Graph.key2node[key]
        node.objects.append(0)"""
