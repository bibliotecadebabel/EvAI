from utilities.Abstract_classes.AbstractSelector import Selector, Observation
import random
import DNA_graph_functions as Funct
import numpy as np

class centered_random_selector(Selector):
    class Observation_creator(Observation):
        def __init__(self,time=0,path=[],weight=0,
            num_layer=1):
            super().__init__(time=time,path=path,weight=weight)
            self.num_layer=num_layer
    def __init__(self,num_actions=5,directions=(
        (0,1,0,0),(0,-1,0,0),
        (1,0,0,0),(-1,0,0,0),
        (0,0,1,1),(0,0,-1,-1),
        (0,0,1),(0,0,-1),
        )):
        super().__init__(self.Observation_creator)
        self.directions=directions
        self.num_actions=num_actions
        self.max_observation_size = 4
        self.current_num_layer=None

    def Action2tensor(self,action):
        pass

    def create_net(self):
        pass

    def comput_max_sub_node(space,sub_nodes):
        particles=[space.node2particles(node) for node in sub_nodes]
        M_index=particles.index(min(particles))
        return sub_nodes[M_index]

    def node2obs(self,space,node):
        creator=self.observation_creator
        weight=Funct.node2num_particles(node)
        path=list(Funct.node2direction(node))
        path[0]=path[0]-self.center
        num_conv_layers = len([0 for layer in space.node2key(node)
            if layer[0] == 0])
        return creator(num_layer=num_conv_layers,
            path=path,weight=weight)

    def register_observations(self, space):
        if type(space) is tuple:
            num_conv_layers = len([0 for layer in space if
                layer[0] == 0])
            self.current_num_layer=num_conv_layers
            self.observations.append(num_layer)
        else:
            center=space.center
            self.current_num_layer=len([0 for layer in center if
                layer[0] == 0])
            node_c=space.key2node(center)
            self.observations=(self.observations+
                [self.node2obs(space,node)
                for node in node_c.kids
                if Funct.node2num_particles(node)>0])

    def update_current_center(self):
        lef_weight=sum([observation.weight for
            observation  in self.observations
            if observation.path[0]<0])
        right_weight=sum([observation.weight for
            observation in self.observations
            if observation.path[0]>0])
        center_weight=sum([observation.weight for
            observation in self.observations
            if observation.path[0]==0])
        if center_weight>max(lef_weight,center_weight):
            pass
        elif right_weight>lef_weight:
            self.center=self.center+1
        elif right_weight<lef_weight:
            self.center=self.center-1


    def print_observation(self):
        if self.observations:
            for observation in self.observations:
                print(f'The weight of {observation.path} '+
                    'is '+ f'{observation.weight}' )
        else:
            print(None)

    def space2action(self,space):
        return None

    def forget_weight(self,observation):
        pass

    def train(self):
        pass

    def update_predicted_actions(self):
        self.predicted_actions=[]
        if self.observations:
            observation=self.observations[0]
            num_layer=observation.num_layer
            num_directions=len(self.directions)
            k=0
            while k<self.num_actions:
                layer=int(np.random.normal(0, 1))+self.center
                if layer>-1 and layer<self.current_num_layer+1:
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
