from abc import ABC, abstractmethod


class Observation(ABC):
    def __init__(self,time=0,path=[],weight=0):
        self.path = path
        self.weight = weight
        self.time = time


class Selector(ABC):

    def __init__(self, Observation_creator: Observation):
        self.observation_creator = Observation_creator
        self.observations = []
        self.actions = []
        self.predicted_actions = []
        self.net = None
        self.current_path = []
        self.max_path_size = 4
        self.max_observation_size = 20
        self.current_time = 0
        self.training_time = 200
        self.dt = 0.01


    def update_current_path(self,space):
        self.current_path.append(space2action(space))
        if len(self.current_path)>self.max_path_size:
            self.current_path.pop(0)
        pass

    def forget_path(self, path):
        if len(path) > self.max_path_size:
            path.pop(0)
        pass

    def forget_observation(self,observation):
        path=observation.path
        self.forget_path(path)
        self.forget_weight(observation)
        pass

    def forget_observations(self):
        while len(self.observations)>self.max_observation_size:
            self.observations.pop(0)
        for observation in self.observations:
            self.observation(observation)
        pass

    def update(space):
        self.register_observations(space)
        self.update_current_path(space)
        self.forget_observations()
        self.train()
        self.update_predicted_actions()
        self.current_time = self.current_time+1
        pass

    #should create net given hyperparameters
    @abstractmethod
    def create_net(self):
        pass

    # The method below should produce the tensorization of a given action
    @abstractmethod
    def Action2tensor(self,action):
        pass

    # The method below, uses the reaction of the given space in response to
    # the selected actions to record new weights
    @abstractmethod
    def register_observations(self, space):
        pass

    @abstractmethod
    def space2action(self,space):
        pass

    @abstractmethod
    def forget_weight(self,observation):
        pass


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def update_predicted_actions(self):
        pass


    #should return predicted actions on suitable format
    @abstractmethod
    def get_predicted_actions(self):
        pass
