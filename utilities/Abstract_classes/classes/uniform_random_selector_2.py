from utilities.Abstract_classes.AbstractSelector import Selector, Observation
import random
import DNA_graph_functions as Funct
import numpy as np
from DNA_conditions import max_layer,max_filter




class centered_random_selector(Selector):
    class Observation_creator(Observation):
        def __init__(self,time=0,path=[],weight=0,
            num_layer=1):
            super().__init__(time=time,path=path,weight=weight)
            self.num_layer=num_layer
    def __init__(self,num_actions=5,directions=None,
        condition=None,
        mutations=(
        (0,1,0,0),(0,-1,0,0),
        (1,0,0,0),(4,0,0,0),
        (0,0,1,1),(0,0,-1,-1),
        (0,0,1),(0,0,-1),
        )):
        print('The condition is')
        print(condition)
        super().__init__(self.Observation_creator)
        self.mutations=mutations
        self.num_actions=num_actions
        self.max_observation_size = 5
        self.current_num_layer=None
        self.version = directions
        if directions=='dupiclate':
            from DNA_directions_duplicate import directions
            self.directions=directions
        elif directions=='clone':
            from DNA_directions_clone import directions as directions
            self.directions=directions
        elif directions=='pool':
            from DNA_directions_pool import directions as directions
            self.directions=directions
        elif directions=='h':
            from DNA_directions_h import directions as directions
            self.directions=directions
        else:
            from DNA_directions_f import directions as directions
            self.directions=directions
        self.center_key=None
        self.condition=condition



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

    def register_observations(self,space,new_center=None):
        creator=self.observation_creator
        if type(space) is tuple:
            num_conv_layers = len([0 for layer in space if
                layer[0] == 0])
            self.current_num_layer=num_conv_layers
            if new_center:
                self.center_key=new_center
            else:
                self.center_key=space
        else:
            if new_center:
                self.center_key=new_center
            else:
                self.center_key=space.center
            center=space.center
            self.current_num_layer=len([0 for layer in new_center if
                layer[0] == 0])
            node_c=space.key2node(center)
            self.observations=(self.observations+
                [self.node2obs(space,node)
                for node in node_c.kids
                if Funct.node2num_particles(node)>0])

    def update_current_center(self,space=None,new_center=None):

        if not(type(space)) is tuple:
            node_nc=space.key2node(new_center)
            if not(new_center==space.center):
                direction=Funct.node2direction(node_nc)
                if direction[1]==(1,0,0,0):
                    self.center=self.center+1
                elif direction[1]==(-1,0,0,0):
                    self.center=self.center-1


        if self.observations:
            node_new_c=space.graph.node2key
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
        num_layer=self.current_num_layer
        num_mutations=len(self.mutations)
        k=0
        l=0
        print(self.mutations)
        while len(self.predicted_actions)<self.num_actions and l<300:
            #layer=int(np.random.normal(0, 3*self.current_num_layer))+self.center
            layer=random.randint(0,self.current_num_layer+1)
            if layer>-1 and layer<self.current_num_layer+1:
                mutation=random.randint(0,num_mutations-1)

                if self.mutations[mutation] == (0, 0, 1):
                    self.__dendrites_mutation(mutation)
                else:
                    DNA=self.center_key
                    condition=self.condition
                    new_DNA=self.directions.get(self.mutations[mutation])(
                        layer,DNA)
                    new_DNA=condition(new_DNA)
                    if  (not ([layer,mutation] in
                        self.predicted_actions) and new_DNA):
                        self.predicted_actions.append([layer,mutation])
                k=k+1
            l=l+1

        print("predicted: ", self.predicted_actions)

    def __dendrites_mutation(self, mutation):

        stop = False
        index_list = []

        for layer_index in range(self.current_num_layer):
            index_list.append(layer_index)

        while len(index_list) > 0 and stop == False:

            random_index = random.randint(0, len(index_list)-1)
            layer = index_list[random_index]
            del index_list[random_index]

            i = 0
            while stop == False and i < 200:

                DNA=self.center_key
                condition=self.condition
                new_DNA=self.directions.get(self.mutations[mutation])(
                    layer,DNA)
                new_DNA=condition(new_DNA)

                if  (not ([layer,mutation] in
                    self.predicted_actions) and new_DNA):
                    self.predicted_actions.append([layer,mutation])
                    stop = True

                i += 1




    def get_predicted_actions(self):
        return tuple([(action[0],self.mutations[action[1]])
        for action in self.predicted_actions])

    def DNA2new_DNA(self,DNA):
        directions=self.directions
        self.update_predicted_actions()
        actions=self.get_predicted_actions()
        direction=actions[0]
        mutation=directions.get(direction[1])
        layer=direction[0]
        return mutation(layer,DNA)




    """def charge_node(self,key,charger=None):
        node=self.Graph.key2node[key]
        node.objects.append(0)"""
