from DNA_Graph import DNA_Graph
from DNA_conditions import max_layer
from DNA_creators import Creator
from DNA_creators import Creator_from_selection as Creator_s
from DNA_creators import Creator_from_selection_nm as Creator_nm
from utilities.Abstract_classes.classes.random_selector import random_selector
from utilities.Abstract_classes.classes.uniform_random_selector import(
    centered_random_selector as Selector_creator)
from DNA_conditions import max_layer,max_filter,max_filter_dense

def DNA_h_nm(x,y):
    max_layers=10
    max_filters=60
    max_dense=100
    def condition_b(z):
        return max_filter_dense(max_filter(max_layer(z,max_layers),max_filters),max_dense)
    center=((-1,1,3,32,32),
            (0,3, 15, 3 , 3),
            (0,18, 15, 3,  3),
            (0,33,33,2,2),
            (0,33, 50, 32, 32),
            (1, 50,10),
             (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4),
            (3,4,5))
    version='h'
    mutations=((4,0,0,0),(1,0,0,0),(0,0,1))
    #mutations=((4,0,0,0),(1,0,0,0),(0,1,0,0),(0,1,0,0),(0,0,1))
    sel=Selector_creator(condition=condition_b,
        directions=version,mutations=mutations,num_actions=10)
    print('The selector is')
    print(sel)
    sel.update(center)
    actions=sel.get_predicted_actions()
    creator=Creator_nm
    space=DNA_Graph(center,1,(x,y),condition_b,actions,
        version,creator=creator,num_morphisms=5,selector=sel)
    space.imprimir()

def DNA_h(x,y):
    max_layers=10
    max_filters=60
    max_dense=100
    def condition_b(z):
        return max_filter_dense(max_filter(max_layer(z,max_layers),max_filters),max_dense)
    center=((-1,1,3,32,32),
            (0,3, 15, 3 , 3),
            (0,18, 15, 3,  3),
            (0,33,33,2,2),
            (0,33, 50, 32, 32),
            (1, 50,10),
             (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4),
            (3,4,5))
    version='h'
    mutations=((4,0,0,0),(1,0,0,0),(0,0,1))
    #mutations=((4,0,0,0),(1,0,0,0),(0,1,0,0),(0,1,0,0),(0,0,1))
    selector=Selector_creator(condition=condition_b,
        directions=version,mutations=mutations,num_actions=10)
    selector.update(center)
    actions=selector.get_predicted_actions()
    creator=Creator_s
    space=DNA_Graph(center,1,(x,y),condition_b,actions,
        version,creator)
    space.imprimir()



def DNA_pool(x,y):
    max_layers=10
    max_filters=60
    max_dense=100
    def condition_b(z):
        return max_filter_dense(max_filter(max_layer(z,max_layers),max_filters),max_dense)
    center=((-1,1,3,x,y),
            (0,3, 15, 3 , 3),
            (0,18, 15, 3,  3),
            (0,33,33,2,2,2),
            (0,33, 50, 13, 13),
            (1, 50,10),
             (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4),
            (3,4,5))
    version='pool'
    mutations=((4,0,0,0),(1,0,0,0),(0,1,0,0),(0,1,0,0))
    selector=Selector_creator(condition=condition_b,
        directions=version,mutations=mutations,num_actions=10)
    selector.update(center)
    actions=selector.get_predicted_actions()
    creator=Creator_s
    space=DNA_Graph(center,1,(x,y),condition_b,actions,
        version,creator)
    space.imprimir()

def DNA_Creator_s(x,y):
    def condition(DNA):
        output=True
        if DNA:
            for num_layer in range(0,len(DNA)-3):
                layer=DNA[num_layer]
                x_l=layer[3]
                y_l=layer[4]
                output=output and (x_l<x) and (y_l<y)
        if output:
            return max_layer(DNA,3)
    center=((0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y), (1, 5, 2), (2,))
    selector=random_selector()
    selector.update(center)
    actions=selector.get_predicted_actions()
    version='inclusion'
    space=DNA_Graph(center,2,(x,y),condition,actions
        ,version,Creator_s)
    space.imprimir()
    print(space.length())


def DNA_test_f(x,y):
    def condition(DNA):
        return max_layer(DNA,15)
    center=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0,5, 5, x-2, y-2),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3))
    version='final'
    space=space=DNA_Graph(center,1,(x,y),condition
        ,((0,0,1),(0,0,1,1),(0,1,0,0),(1,0,0,0)),version)
    space.imprimir()
    print(space.length())



def DNA_test_i(x,y):
    def condition(DNA):
        output=True
        if DNA:
            for num_layer in range(0,len(DNA)-3):
                layer=DNA[num_layer]
                x_l=layer[3]
                y_l=layer[4]
                output=output and (x_l<x) and (y_l<y)
        if output:
            return max_layer(DNA,3)
    center=((0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y), (1, 5, 2), (2,))
    version='inclusion'
    space=space=DNA_Graph(center,1,(x,y),condition,(0,(0,0,1,1),
        (0,1,0,0),(1,0,0,0)),version)
    space.imprimir()
    print(space.length())


def layer_increase_i(x,y):
    def condition(DNA):
        output=True
        if DNA:
            for num_layer in range(0,len(DNA)-3):
                layer=DNA[num_layer]
                x_l=layer[3]
                y_l=layer[4]
                output=output and (x_l<x) and (y_l<y)
        if output:
            return max_layer(DNA,10)
    center=((0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0, 7,5, x, y), (1, 5, 2), (2,))
    version='inclusion'
    space=space=DNA_Graph(center,4,(x,y),condition,(0,(-1,0,0,0)),version)
    space.imprimir()
    print(space.length())

def kernel_increase_i(x,y):
    def condition(DNA):
        output=True
        if DNA:
            for num_layer in range(0,len(DNA)-3):
                layer=DNA[num_layer]
                x_l=layer[3]
                y_l=layer[4]
                output=output and (x_l<x) and (y_l<y)
        if output:
            return max_layer(DNA,10)
    center=((0, 3, 5, 3, 3),(0, 3, 8, 3,3),(0, 7,5, x, y), (1, 5, 2), (2,))
    version='inclusion'
    space=space=DNA_Graph(center,4,(x,y),condition,(0,(0,0,1,1)),version)
    space.imprimir()
    print(space.length())


def add_filter_i(x,y):
    def condition(DNA):
        return max_layer(DNA,10)
    center=((0, 3, 4, 2, 2),(0,7,4, x, y), (1, 4, 2), (2,))
    space=DNA_Graph(center,2,(x,y),condition,(0,(0,1,0,0)))
    space.imprimir()
    #print(space.key2node(center))
    return space


def kernel_height_creator_i(x,y):
    def condition(DNA):
        return max_layer(DNA,10)
    creator=Creator(((0,1,0,0),(0,0,1,1)),condition)
    center=((0, 3, 4, 2, 2),(0, 7,5, x-1, y-1), (1, 5, 2), (2,))
    space=space=DNA_Graph(center,20,(x,y),condition,(0,(0,0,1,1)))
    print('linear_filter: done')
    space.imprimir()
    print(space.length())


def kernel_height_creator(x,y):
    def condition(DNA):
        return max_layer(DNA,10)
    creator=Creator(((0,1,0,0),(0,0,1,1)),condition)
    center=((0, 3, 4, 2, 2),(0, 4,5, x-1, y-1), (1, 5, 2), (2,))
    space=space=DNA_Graph(center,20,(x,y),condition,(0,(0,0,1,1)))
    print('linear_filter: done')
    space.imprimir()
    print(space.length())

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
    space=space=DNA_Graph(center,20,(x,y),condition,(0,(0,0,1,1)))
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
    space=DNA_Graph(center,2,(x,y),(0,(0,0,0,1,1)))
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


DNA_h_nm(11,11)
#DNA_test_f(11,11)
#DNA_pool(11,11)
#DNA_h(11,11)
#DNA_Creator_s(11,11)
#DNA_test_i(11,11)
#layer_increase_i(11,11)
#kernel_increase_i(11,11)
#add_filter_i(11,11)
#linear_filter_creator(11,11)
#linear_filter(11,11)
#   linear_kernel_depth(11,11)
#kernel_height_creator(11,11)
#kernel_height_creator_i(11,11)
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
