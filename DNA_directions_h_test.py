import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane
import DNA_directions_h as dire
import test_DNAs as DNAs
import DNA_conditions


def add_conv_last():
    dict={DNA_conditions.max_filter : 256,
            DNA_conditions.max_filter_dense : 530,
            DNA_conditions.max_kernel_dense : 9,
            DNA_conditions.max_layer : 30,
            DNA_conditions.min_filter : 3,
            DNA_conditions.max_pool_layer : 5,
            DNA_conditions.max_parents : 2}
    def condition(DNA):
        return DNA_conditions.dict2condition(DNA,dict)
    DNA=DNAs.DNA_ep_85_5_pool
    DNA=dire.add_layer(26,DNA)
    print(condition(DNA))

    print(dire.initialize_kernel())

def initialize_kernel_test():
    print('The new dimension is')
    print(dire.initialize_kernel())

def Persistent_synapse_condition_test(x,y):
    DNA = ((-1,1,3,11,11),(0, 3, 5, 3, 3),(0, 8, 8,3,3),
             (0, 8, 8,7,7),
             (1, 7, 2), (2,),(3,-1,0),(3,0,1),
             (3,1,2),(3,2,3),(3,3,4))
    print(dire.Persistent_synapse_condition(DNA))

def compute_num_layers_test(x,y):
    DNA=((-1,1,3,11,11),(0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    print('The number of layers in')
    print(DNA)
    print('is')
    print(len(dire.DNA2layers(DNA)))

def DNA2layers_test(x,y):
    DNA=((-1,1,3,11,11),(0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    DNA_p=((-1,1,3,11,11),(0, 3, 5, 3, 3),(0, 5, 8, 3,3),(4,13,13,2,2),(0,13,5, 4, 4),
            (1, 5, 2),
            (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,0,2),
            (3,2,3),
            (3,3,4),
            (3,4,5))
    print('The layers are')
    print(dire.DNA2layers(DNA_p))

def DNA2synapses_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, 7, 7),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    print('The synases are')
    print(dire.DNA2synapses(DNA))

def DNA2graph_test(x,y):
    DNA=((-1,1,3,11,11),(0, 3, 5, 3, 3),(0, 5, 8, 3,3),(4,13,13,2,2),(0,13,5, 4, 4),
            (1, 5, 2),
            (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,0,2),
            (3,2,3),
            (3,3,4),
            (3,4,5))
    print('The graph is')
    dire.imprimir(dire.DNA2graph(DNA))

def compute_output_test():
    DNA=((-1,1,3,11,11),(0, 3, 5, 3, 3),(0, 5, 5, 3,3),(0,5,13,2,2,2),
            (1, 13, 2),
            (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4))
    g=dire.DNA2graph(DNA)
    full_node=dire.graph2full_node(g)
    dire.compute_output(g,full_node)
    dire.imprimir(g)

def DNA2graph_relable_test(x,y):
    def map(k):
        return k+1
    DNA=((-1,11,11),(0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    print('The graph is')
    g=dire.DNA2graph(DNA)
    g.relable(map)
    dire.imprimir(g)

def retract_h_test():
    DNA=DNAs.DNA_h
    DNA=dire.retract_dendrites(4,DNA)
    print(DNA)

def graph2DNA_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, 7, 7),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    print('The DNA is')
    print(dire.graph2DNA(dire.DNA2graph(DNA)))

def increase_filters_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0,5, 5, x-2, y-2),
            (1, 5, 10), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3))
    print('The DNA is')
    print(dire.increase_filters(0,DNA))

def decrease_filters_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 12, 3, 3),(0,12, 5, x-2, y-2),
            (1, 5, 10), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3))
    while DNA:
        print('The DNA is')
        DNA=dire.decrease_filters(0,DNA)
        print(DNA)



def remove_layer_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 3,3),(0,13,5, 7, 7),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,-1,2),
            (3,1,2),(3,2,3),(3,3,4))
    #while DNA:
    print('The new DNA is')
    while DNA:
        DNA=dire.remove_layer(0,DNA)
        print(DNA)

def add_layer_test(x,y):
    DNA=((-1,1,3,11,11),(0, 3, 5, 3, 3),(0, 5, 8,2,2,2),(0,8,120,6,6),
            (1, 13, 2),
            (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4))
    g=dire.DNA2graph(DNA)
    print('The old DNA is')
    print(DNA)
    #while DNA:
    print('The new DNA is')
    DNA=dire.add_layer(0,DNA)
    print(DNA)
    #g=dire.DNA2graph(DNA)
    #dire.compute_output(g)
    #dire.imprimir(g)



def add_pool_layer_test(x,y):
    DNA=((-1,1,3,11,11),(0, 3, 5, 3, 3),(0, 5, 8,2,2,2),(0,8,120,4,4),
            (1, 13, 2),
            (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4))
    g=dire.DNA2graph(DNA)
    print('The old DNA is')
    print(DNA)
    #while DNA:
    print('The new DNA is')
    DNA=dire.add_pool_layer(2,DNA)
    print(DNA)
    #g=dire.DNA2graph(DNA)
    #full_node=dire.graph2full_node(g)
    #dire.compute_output(g,full_node)
    #dire.imprimir(g)



def remove_layer_test_2(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(4,5,5,2,2),(0, 5, 5, 4, 4),
            (1, 5, 2), (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4))
    print('old DNA is:')
    print(DNA)
    print('The new DNA is')
    while DNA:
        DNA=dire.remove_layer(1,DNA)
        print(DNA)

def fix_fully_conected_test(x,y):
    DNA=((-1,1,3,11,11),(0, 3, 5, 3, 3),(0, 5, 5, 3,3),(0,5,13,5,5,2),
            (1, 13, 2),
            (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4))
    g=dire.DNA2graph(DNA)
    dire.fix_fully_conected(g)
    dire.imprimir(g)

def increase_kernel_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 7,7),(0,13,5, 2, 2),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4))
    #while DNA:
    print('The new DNA is')
    print(dire.increase_kernel(1,DNA))

def decrease_kernel_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 7,7),(0,13,5, 2, 2),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4))
    #while DNA:
    print('The new DNA is')
    print(dire.decrease_kernel(1,DNA))

def spread_dendrites_test_1(x,y):
    DNA=((-1, 1, 3, 32, 32), (0, 3, 128, 4, 4), (0, 131, 128, 5, 5, 2), (0, 128, 128, 7, 7, 2), (0, 128, 128, 3, 3), (0, 256, 64, 8, 8), (1, 64, 10), (2,), (3, -1, 0), (3, 0, 1), (3, -1, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 2, 4), (3, 4, 5), (3, 5, 6))

    #while DNA:
    print('The old DNA is')
    print(DNA)
    print('The new DNA is')
    print(dire.spread_dendrites(1 ,DNA))



def spread_dendrites_test_2(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 7,7),(0,8,5, 2, 2),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),(3,-1,1),
            (3,1,2),(3,0,2),(3,2,3),(3,3,4))
    print('The new DNA is')
    print(dire.spread_dendrites(0,DNA))

def spread_dendrites_test_3(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 7,7),(0,8,5, 2, 2),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,0,2),(3,-1,2),(3,2,3),(3,3,4))
    print('The new DNA is')
    print(dire.spread_dendrites(0,DNA))

def spread_dendrites_test_4(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 7,7),(0,8,5, 2, 2),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),(3,-1,1),
            (3,1,2),(3,0,2),(3,-1,2),(3,2,3),(3,3,4))
    print('The new DNA is')
    print(dire.spread_dendrites(0,DNA))

def spread_dendrites_test_5(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 8, 5, 3,3),(0,8,5, 3, 3),
            (0,5,5, 3, 3),
            (1, 5, 2), (2,),
            (3,-1,0),
            (3,0,1),
            (3,-1,1),
            (3,1,2),
            (3,-1,2),
            (3,2,3),
            (3,3,4),
            (3,4,5))
    print('The new DNA is')
    print(dire.spread_dendrites(0,DNA))

def spread_dendrites_test_6(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 2, 2),(0, 5, 5, 2,2),
            (0,5,5, 2, 2),
            (0,5,5, 2, 2,2),
            (0,5,5, 2, 2),
            (0,5,5, 2, 2,2),
            (0,5,5, 3, 3),
            (1, 5, 2), (2,),
            (3,-1,0),
            (3,0,1),
            (3,1,2),
            (3,2,3),
            (3,3,4),
            (3,4,5),
            (3,5,6),
            (3,6,7),
            (3,7,8))
    print('The new DNA is')
    k=0
    while k<5 and DNA:
        DNA=dire.spread_dendrites(0,DNA)
        print(DNA)
        k=k+1
        print(k)

def retract_dendrites_test(x,y):
    print('The old DNA is')
    DNA=((-1, 1, 3, 32, 32), (0, 3, 128, 4, 4), (0, 131, 128, 5, 5, 2), (0, 128, 128, 7, 7, 2), (0, 128, 128, 3, 3), (0, 256, 64, 8, 8), (1, 64, 10), (2,), (3, -1, 0), (3, 0, 1), (3, -1, 1), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 2, 4), (3, 4, 5), (3, 5, 6))
    print(DNA)
    print('The new DNA is')
    DNA=dire.retract_dendrites(0,DNA)
    print(DNA)


add_conv_last()
#initialize_kernel_test()
#remove_layer_test_2(11,11)
#add_pool_layer_test(11,11)
#compute_output_test()
#retract_dendrites_test(11,11)
#spread_dendrites_test_6(32,32)
#spread_dendrites_test_5(11,11)
#spread_dendrites_test_4(11,11)
#spread_dendrites_test_3(11,11)
#spread_dendrites_test_1(11,11)
#increase_kernel_test(11,11)
##decrease_kernel_test(11,11)
#Persistent_synapse_condition_test(11,11)
#add_layer_test(11,11)
#add_layer_test(11,11)
#retract_h_test()
#add_pool_layer_test(32,32)
#fix_fully_conected_test(11,11)
#compute_num_layers_test(11,11)
#remove_layer_test(11,11)
##DNA2graph_relable_test(11,11)
#decrease_filters_test(11,11)
#increase_filters_test(11,11)
#graph2DNA_test(11,11)
#DNA2graph_test(11,11)
#DNA2synapses_test(11,11)
#DNA2layers_test(11,11)
