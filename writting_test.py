import FileController.Controller as FileController

from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
from DNA_Phase_space import DNA_Phase_space
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
    Phase_space.print_diffussion_field()

def update_interaction_field():
    Phase_space=initialize()
    Phase_space.create_particles(100)
    Phase_space.update_density()
    Phase_space.alpha=1.5
    Phase_space.update_interaction_field()
    print('update_diffussion_field: done')
    
    return Phase_space.DNA_graph

def update_external_field():
    Phase_space=initialize()
    Phase_space.create_particles(100)
    Phase_space.update_density()
    Phase_space.update_external_field()
    print('update_external_field: done')
    Phase_space.print_external_field()




space = update_interaction_field()
controller = FileController.Controller(space)
controller.imprimir(2, "TESTING-DNA")

