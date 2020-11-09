from utilities.Abstract_classes.AbstractSelector import Selector, Observation
import random
import Geometric.Graphs.DNA_graph_functions as Funct
import numpy as np
import LSTM.network_lstm as net_lstm
import utilities.lstm_converter as LSTMConverter
from Geometric.Conditions.DNA_conditions import max_layer,max_filter
import time
import const.general_values as const_values



class LSTMSelector(Selector):
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
        self.current_num_layer=None
        self.version = directions
        
        if directions=='dupiclate':
            from Geometric.Directions.DNA_directions_duplicate import directions
            self.directions=directions
        elif directions=='clone':
            from Geometric.Directions.DNA_directions_clone import directions as directions
            self.directions=directions
        elif directions=='pool':
            from Geometric.Directions.DNA_directions_pool import directions as directions
            self.directions=directions
        elif directions=='pool_duplicate':
            from Geometric.Directions.DNA_directions_pool_duplicate import directions as directions
            self.directions=directions
        elif directions=='h':
            from Geometric.Directions.DNA_directions_h import directions as directions
            self.directions=directions
        elif directions=='convex':
            from Geometric.Directions.DNA_directions_convex import directions as directions
            self.directions=directions
        else:
            from Geometric.Directions.DNA_directions_f import directions as directions
            self.directions=directions
        self.center_key=None
        self.condition=condition
        
        self.max_layers_condition = 20
        self.max_layers_lstm = self.max_layers_condition*2 #positive and negative relative position
        self.max_observation_size = 3
        self.max_path_size = self.max_observation_size-1

        self.converter = LSTMConverter.LSTMConverter(cuda=True, max_layers=self.max_layers_lstm, mutation_list=self.mutations,
                                                        limit_directions=self.max_observation_size)

        self.net = net_lstm.NetworkLSTM(observation_size=self.max_observation_size, 
                                            in_channels=self.max_layers_lstm, 
                                            out_channels=self.max_layers_lstm*self.converter.mutations, 
                                            kernel_size=self.converter.mutations, cuda_flag=True)
        self.input_lstm = None
        self.current_path = self.observation_creator(num_layer=0,
            path=[const_values.EMPTY_DIRECTION], weight=1)
        self.current_path.absolute_path.append(const_values.EMPTY_DIRECTION)
        
        


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
            
            self.observations = []
            
            weight_array = []
            i = 0
            for node in node_c.kids:

                if Funct.node2energy(node) > 0:
                    weight_array.append(Funct.node2energy(node)*10)

            weight_array = Funct.normalize(dataset=weight_array)
            for node in node_c.kids:

                if Funct.node2energy(node) > 0:

                    observation_path = []
                    for direction in self.current_path.path:
                        observation_path.append(direction)

                    absolute_direction = Funct.node2direction(node)
                    relative_direction = Funct.absolute2relative(direction=absolute_direction, last_layer=self.last_mutated_layer, 
                                                                    max_layers=self.max_layers_condition)
                    observation_path.append(relative_direction)

                    if len(observation_path) > self.max_observation_size:
                        observation_path.pop(0)
                    
                    observation = self.observation_creator(path=observation_path, weight=weight_array[i])
                    self.observations.append(observation)

                    i += 1

    def update_current_center(self,space=None,new_center=None):

        if type(space) is not tuple:

            #node_nc = space.key2node(new_center)
            direction = self.current_path.absolute_path[-1]

            if not(new_center==space.center) and direction is not None and direction[0] != const_values.EMPTY_INDEX_LAYER:

                self.last_mutated_layer = direction[0]
                print("last layer mutated: ", self.last_mutated_layer)
            


    def print_observation(self):
        if self.observations:
            for observation in self.observations:
                print(f'The weight of {observation.path} '+
                    'is '+ f'{observation.weight}' )
        else:
            print(None)

    def space2action(self,space):
        
        return None

    def forget_filters(self,observation):
        pass

    def update_current_path(self,space, new_center):
        
        if not isinstance(space, tuple) and new_center is not None:
            node_nc = space.key2node(new_center)
            absolute_direction = Funct.node2direction(node_nc)
            relative_direction = Funct.absolute2relative(direction=absolute_direction, last_layer=self.last_mutated_layer,
                                                            max_layers=self.max_layers_condition)
            self.current_path.path.append(relative_direction)
            self.current_path.absolute_path.append(absolute_direction)

            if len(self.current_path.path)>self.max_path_size:
                self.current_path.path.pop(0)
                self.current_path.absolute_path.pop(0)

    def train(self):
        
        if self.observations is not None and len(self.observations) > 0:

            for i in range(len(self.observations)):
                print("i: ", i)
                print("path: ", self.observations[i].path)
                print("weight: ", self.observations[i].weight)

            print("current path: ", self.current_path.path)                
            input_lstm = self.converter.generate_LSTM_input(observations=self.observations)
            print("starting lstm training")
            print("input lstm: ", input_lstm.size())
            self.net.Training(data=input_lstm, observations=self.observations, dt=self.dt, p=self.training_time)

    def update_predicted_actions(self):

        self.predicted_actions=[]
        self.__run_predict()
        #print("predicted actions 1: ", self.predicted_actions)
        num_layer=self.current_num_layer
        num_mutations=len(self.mutations)
        k=0
        l=0
        #print(self.mutations)
        while len(self.predicted_actions)<self.num_actions and l<300:
            #layer=int(np.random.normal(0, 3*self.current_num_layer))+self.center
            layer=random.randint(0,self.current_num_layer+1)
            if layer>-1 and layer<self.current_num_layer+2:
                mutation=random.randint(0,num_mutations-1)

                if self.mutations[mutation] == (0, 0, 1) or self.mutations[mutation] == (0, 0, 2):
                    self.__dendrites_mutation(mutation)
                elif self.mutations[mutation] == (1, 0, 0, 0):
                    self.__addLayer_convex(mutation)
                else:
                   
                    direction_accepted = self.verify_direction(layer=layer, mutation=self.mutations[mutation])

                    if direction_accepted == True:
                        self.predicted_actions.append( (layer,self.mutations[mutation]) )
                k=k+1
            l=l+1
        print("predicted actions: ", self.predicted_actions)
        time.sleep(5)

    def __run_predict(self):

        if self.observations is not None and len(self.observations) > 0:
            
            input_predict = self.converter.generate_LSTM_predict(observation=self.current_path)
            top_directions = len(self.observations) // 2
            predicted_tensor = self.net.predict(x=input_predict)
            predicted_relative_directions = self.converter.topK_predicted_directions(predicted_tensor=predicted_tensor, k=len(self.observations))

            print("relative predicted: ", predicted_relative_directions)
            predicted_absolute_directions = []

            for relative_direction in predicted_relative_directions:
                absolute_direction = Funct.relative2absolute(direction=relative_direction, last_layer=self.last_mutated_layer,
                                                                max_layers=self.max_layers_condition)
                predicted_absolute_directions.append(absolute_direction)

            for direction in predicted_absolute_directions:
                
                layer = direction[0]
                mutation = direction[1]

                direction_accepted = self.verify_direction(layer=layer, mutation=mutation)

                if direction_accepted == True:  
                    self.predicted_actions.append( (layer,mutation) )
                
                if len(self.predicted_actions) >= top_directions:
                    break
            
            print("predicted by lstm: ", self.predicted_actions)

    def __dendrites_mutation(self, mutation):

        stop = False
        index_list = []

        for layer_index in range(self.current_num_layer-2):
            index_list.append(layer_index)

        while len(index_list) > 0 and stop == False:

            random_index = random.randint(0, len(index_list)-1)
            layer = index_list[random_index]
            del index_list[random_index]

            i = 0
            while stop == False and i < 200:

                direction_accepted = self.verify_direction(layer=layer, mutation=self.mutations[mutation])

                if direction_accepted == True:
                    self.predicted_actions.append( (layer,self.mutations[mutation]) )
                    stop = True

                i += 1

    def __addLayer_convex(self, mutation):
        
        graph_dna = Funct.DNA2graph(self.center_key)

        parent_layers = []

        acum_parents = 0
        acum_history = []

        for layer_index in range(self.current_num_layer):
            
            node = graph_dna.key2node.get(layer_index)
            parents = len(node.parents)
            parent_layers.append(parents)
            acum_parents += parents
            acum_history.append(acum_parents)

        factor = random.randint(0, sum(parent_layers))
        selected_index = -1

        for i in range(len(acum_history)):

            if factor <= acum_history[i]:
                selected_index = i
                break
        
        
        direction_accepted = self.verify_direction(layer=selected_index, mutation=self.mutations[mutation])

        if direction_accepted == True:
            self.predicted_actions.append( (selected_index, self.mutations[mutation]) )

    def verify_direction(self, layer, mutation):

        try:

            value = False

            if layer >= 0 and layer < self.current_num_layer:

                DNA=self.center_key
                condition=self.condition
                new_DNA=self.directions.get(mutation)(layer, DNA)
                new_DNA=condition(new_DNA)

                if  (not ( (layer, mutation) in self.predicted_actions) and new_DNA):
                    value = True
                
                if mutation == (1,0,0,0) or mutation == (4,0,0,0):

                    if (layer + 1) == self.current_num_layer:
                        value = False
        except:
            print("ERROR MUTATING DNA")
            print("layer: ", layer)
            print("mutation: ", mutation)
            print("last mutated layer: ", self.last_mutated_layer)
            raise
        
        return value

    def get_predicted_actions(self):
        #return tuple([(action[0],self.mutations[action[1]])
        #for action in self.predicted_actions])
        return self.predicted_actions
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
