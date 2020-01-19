from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph




def linear_filter():
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
    print('linear_filter: done')
    space.imprimir()
    print(space.type)
    return space
    #print(space.key2node(center))

def linear_kernel_width():
    S=100
    Comp=2
    dataGen=GeneratorFromImage.GeneratorFromImage(
    Comp, S, cuda=False)
    dataGen.dataConv2d()
    x = dataGen.size[1]
    y = dataGen.size[2]
    ks=[2]
    center=((0, 3, int(2*ks[0]), 2, 2),(0, int(2*ks[0]),ks[0], x-1, y-1), (1, ks[0], 2), (2,))
    print('linear_kernel_width: done')
    space=DNA_Graph(center,5,(x,y),(0,(0,1,0,0)))
    print(space.type)
    space.imprimir()
    #print(space.key2node(center))
    return space

linear_filter()
linear_kernel_width()

#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
