from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
from DNA_Phase_space import DNA_Phase_space
from Dynamic_DNA import Dynamic_DNA
from DNA_conditions import max_layer
from DNA_creators import Creator

def initialize():
    S=100
    Comp=2
    dataGen=GeneratorFromImage.GeneratorFromImage(
    Comp, S, cuda=False)
    dataGen.dataConv2d()
    x = dataGen.size[1]
    y = dataGen.size[2]
    ks=[2]
    def condition(DNA):
        return max_layer(DNA,10)
    center=((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
    space=DNA_Graph(center,2,(x,y),condition,(0,(1,1,0,0)))
    Phase_space=DNA_Phase_space(space)
    Dynamics=Dynamic_DNA(space,Phase_space)
    print('initialized')
    return Dynamics


def update_force_field():
    Dynamics=initialize()
    Phase_space=Dynamics.phase_space
    Phase_space.create_particles(100)
    Phase_space.beta=2
    Phase_space.update_density()
    Phase_space.update_external_field()
    Phase_space.update_diffussion_field()
    Dynamics.update_force_field()
    print('force field updated')
    Phase_space.print_force_field()
    pass

def update_velocity():
    Dynamics=initialize()
    Phase_space=Dynamics.phase_space
    Phase_space.create_particles(100)
    Phase_space.beta=2
    Phase_space.update_density()
    Phase_space.update_external_field()
    Phase_space.update_diffussion_field()
    Dynamics.update_force_field()
    Dynamics.update_velocity()
    print('velocity updated')
    pass

def update_particles():
    Dynamics=initialize()
    Phase_space=Dynamics.phase_space
    Phase_space.create_particles(100)
    Phase_space.beta=2
    Phase_space.update_density()
    Phase_space.update_external_field()
    Phase_space.update_diffussion_field()
    Dynamics.update_force_field()
    Dynamics.update_velocity()
    Dynamics.update_particles()
    print('particles updated')

def print_particles():
    Dynamics=initialize()
    Phase_space=Dynamics.phase_space
    Phase_space.create_particles(100)
    Phase_space.beta=2
    Phase_space.update_density()
    Phase_space.update_external_field()
    Phase_space.update_diffussion_field()
    Dynamics.update_force_field()
    Dynamics.update_velocity()
    Dynamics.update_particles()
    Dynamics.print_particles()


def update():
    Dynamics=initialize()
    Phase_space=Dynamics.phase_space
    Phase_space.create_particles(100)
    Phase_space.beta=2
    Dynamics.update()
    print('Dynamics updated')

#initialize()
#update_external_field()
#update_velocity()
#update_particles()
#update()
update_force_field()
#print_particles()
#create_particles()
#update_density()
#update_divergence()
#update_diffussion_field()
#update_external_field()







#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
