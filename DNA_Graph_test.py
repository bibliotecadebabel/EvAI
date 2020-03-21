from DNA_Graph import DNA_Graph
from DNA_conditions import max_layer
from DNA_creators import Creator


def linear_filter_creator(x,y):
    def condition(DNA):
        return max_layer(DNA,10)
    creator=Creator(((0,1,0,0),(0,0,1,1)),condition)
    center=((0, 3, 4, 2, 2),(0, 4,5, x-1, y-1), (1, 5, 2), (2,))
    space=space=DNA_Graph(center,2,(x,y),condition,((0,1,0,0),(0,0,1,1)))
    print('linear_filter: done')
    space.imprimir()
    print(space.length())

def kernel_height_creator(x,y):
    def condition(DNA):
        return max_layer(DNA,10)
    creator=Creator(((0,1,0,0),(0,0,1,1)),condition)
    center=((0, 3, 4, 2, 2),(0, 4,5, x-1, y-1), (1, 5, 2), (2,))
    space=space=DNA_Graph(center,8,(x,y),condition,(0,(0,0,1,1)))
    print('linear_filter: done')
    space.imprimir()
    print(space.length())

def linear_kernel_depth(x,y):
    def condition(DNA):
        return max_layer(DNA,10)
    center=((0, 3, 10, 1,3, 3),(0,1,20, 3,3,3),(0,1,20,160,7, 7),(1,20 , 2), (2,))
    print('linear_kernel_depth: done')
    space=DNA_Graph(center,5,(x,y),condition,(0,(0,0,1,1)))
    space.imprimir()
    print(space.length())
    return space

def linear_kernel_width_new(x,y):
    ks=[2]
    center=((0, 3, int(2*ks[0]), 1,2, 2),(0, int(2*ks[0]),ks[0], 1,x-1, y-1), (1, ks[0], 2), (2,))
    print('linear_kernel_width_new: done')
    space=DNA_Graph(center,5,(x,y),(0,(0,0,0,1,1)))
    print(space.type)
    space.imprimir()
    #print(space.key2node(center))
    return space

def linear_filter_new(x,y):
    ks=[2]
    center=((0, 3,ks[0], 1,x, y), (1, ks[0], 2), (2,))
    space=DNA_Graph(center,5,(x,y),(0,(0,1,0,0,0)))
    print('linear_filter_new: done')
    space.imprimir()
    print(space.type)
    return space


def linear_filter(x,y):
    ks=[2]
    center=((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
    space=DNA_Graph(center,5,(x,y))
    print('linear_filter: done')
    space.imprimir()
    print(space.type)
    return space
    #print(space.key2node(center))

def linear_kernel_width(x,y):
    ks=[2]
    center=((0, 3, int(2*ks[0]), 2, 2),(0, int(2*ks[0]),ks[0], x-1, y-1), (1, ks[0], 2), (2,))
    print('linear_kernel_width: done')
    space=DNA_Graph(center,5,(x,y),(0,(0,1,0,0)))
    print(space.type)
    space.imprimir()
    #print(space.key2node(center))
    return space


#linear_filter_creator(11,11)
#linear_filter(11,11)
#   linear_kernel_depth(11,11)
kernel_height_creator(11,11)
#linear_filter_new(11,11)
#linear_kernel_width_new(11,11)
#linear_filter(11,11)
#linear_kernel_width(11,11)

#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
