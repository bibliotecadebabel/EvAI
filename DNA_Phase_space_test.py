from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
from DNA_Phase_space import DNA_Phase_space

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
    return Phase_space

def create_particles():
    Phase_space=initialize()
    Phase_space.create_particles(100)
    print('create_particles: done')

def update_density():
    Phase_space=initialize()
    Phase_space.create_particles(100)
    Phase_space.update_density()
    print('update_density: done')

def update_divergence():
    Phase_space=initialize()
    Phase_space.create_particles(100)
    Phase_space.update_density()
    Phase_space.update_divergence()
    print('update_divergence: done')

def update_diffussion_field():
    Phase_space=initialize()
    Phase_space.create_particles(100)
    Phase_space.update_density()
    Phase_space.beta=1.5
    Phase_space.update_diffussion_field()
    print('update_diffussion_field: done')
    pass

def update_external_field():
    Phase_space=initialize()
    Phase_space.create_particles(100)
    Phase_space.update_density()
    Phase_space.update_external_field()
    pass



#create_particles()
#update_density()
#update_divergence()
#update_diffussion_field()
update_external_field()







#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
