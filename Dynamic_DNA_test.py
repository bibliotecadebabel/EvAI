from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
from DNA_Phase_space import DNA_Phase_space
from Dynamic_DNA import Dynamic_DNA

def initialize():
    S=100
    Comp=2
    dataGen=GeneratorFromImage.GeneratorFromImage(
    Comp, S, cuda=False)
    dataGen.dataConv2d()
    x = dataGen.size[1]
    y = dataGen.size[2]
    ks=[2]
    center=((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
    space=DNA_Graph(center,5,(x,y))
    Phase_space=DNA_Phase_space(space)
    Dynamics=Dynamic_DNA(space,Phase_space)
    print('initialized')
    return Dynamics


def update_external_field():
    Dynamics=initialize()
    Phase_space=Dynamics.phase_space
    Phase_space.create_particles(100)
    Phase_space.beta=2
    Phase_space.update_density()
    Phase_space.update_external_field()
    Phase_space.update_diffussion_field()
    Dynamics.update_force_field()
    print('force field updated')
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

#initialize()
#update_external_field()
#update_velocity()
update_particles()
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
