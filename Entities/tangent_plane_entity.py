import json

class TangentPlaneEntity():

    def __init__(self):
        self.divergence=None
        self.metric=None
        self.density=None
        self.num_particles=None
        self.gradient=None
        self.reg_density=None
        self.interaction_field=None
        self.difussion_field=None
        self.external_field=None
        self.force_field=None
        self.energy=None
        self.interaction_potential=None
        self.velocity_potential=None
        self.direction=None
    
    def load(self, data):
        self.divergence=data['divergence']
        self.metric=data['metric']
        self.density=data['density']
        self.num_particles=data['num_particles']
        self.gradient=data['gradient']
        self.reg_density=data['reg_density']
        self.interaction_field=data['interaction_field']
        self.difussion_field=data['difussion_field']
        self.external_field=data['external_field']
        self.force_field=data['force_field']
        self.energy=data['energy']
        self.interaction_potential=data['interaction_potential']
        self.velocity_potential=data['velocity_potential']
        self.direction=data['direction']

        self.divergence = self.__listToTuple(self.divergence)
        self.metric = self.__listToTuple(self.metric)
        self.density = self.__listToTuple(self.density)
        self.gradient = self.__listToTuple(self.gradient)
        self.reg_density = self.__listToTuple(self.reg_density)
        self.interaction_field = self.__listToTuple(self.interaction_field)
        self.difussion_field = self.__listToTuple(self.difussion_field)
        self.external_field = self.__listToTuple(self.external_field)
        self.force_field = self.__listToTuple(self.force_field)
        self.direction = self.__listToTuple(self.direction)

    
    def __listToTuple(self, list_object):
        
        if type(list_object) == list:

            for i in range(len(list_object)):

                if type(list_object[i]) == list:
                    list_object[i] = tuple(list_object[i])
            
            list_object = tuple(list_object)
            
        return list_object

