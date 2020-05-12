from DNA_Graph import DNA_Graph
from DNA_conditions import max_layer
from DNA_creators import Creator_from_selection as Creator_s
from utilities.Abstract_classes.classes.random_selector  import(
    random_selector)
from utilities.Abstract_classes.classes.centered_random_selector import(
    centered_random_selector as Cen_rand_select)
import DNA_graph_functions as Funct




def random_centered_test(x,y):
    def condition(DNA):
        return max_layer(DNA,15)
    center=((-1,1,3,x,y),
            (0,3, 5, x, y),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2))
    version='final'
    space=space=DNA_Graph(center,1,(x,y),condition
        ,((0,0,1),(0,0,1,1),(0,1,0,0),(1,0,0,0)),version)
    center=space.center
    node_c=space.graph.key2node.get(center)
    Funct.set_num_particles(node_c.kids[0],10)
    print(f'the center is {center}')
    Selector=Cen_rand_select()
    k=10
    while k>0:
        Selector.update(space)
        Selector.print_observation()
        print(Selector.get_predicted_actions())
        k=k-1
    actions=Selector.get_predicted_actions()
    space=space=DNA_Graph(center,1,(x,y),condition,
        actions,version,
        Creator_s)
    space.imprimir()




def random_predict_test(x,y):
    selector=random_selector()
    center=((-11,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 3,3),
        (0,8,5, x-4, y-4), (1, 5, 2), (2,))
    selector.update(center)
    print(selector.get_predicted_actions())


def random_forget_test(x,y):
    selector=random_selector()
    center=((-11,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 3,3),
        (0,8,5, x-4, y-4), (1, 5, 2), (2,))
    selector.register_observations(center)
    print('The registered observation is:')
    selector.print_observation()
    selector.forget_observations()
    print('The forgoten observation is:')
    selector.print_observation()



def random_record_test(x,y):
    selector=random_selector()
    center=((-11,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 3,3),
        (0,8,5, x-4, y-4), (1, 5, 2), (2,))
    selector.register_observations(center)
    print('The registered observation is:')
    selector.print_observation()



def random_selector_test():
    selector=random_selector()
    print('Done')
    pass

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
    version='inclusion'
    space=DNA_Graph(center,2,(x,y),condition,((0,(0,0,1,1)),
        (1,(0,1,0,0)))
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
    space=space=DNA_Graph(center,2,(x,y),condition
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

random_centered_test(11,11)
#random_predict_test(11,11)
#random_forget_test(11,11)
#random_record_test(11,11)
#random_selector_test()
#DNA_test_f(11,11)
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
