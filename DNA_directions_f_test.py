import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane
import DNA_directions_f as dire


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
    print('The layers are')
    print(dire.DNA2layers(DNA))

def DNA2synapses_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, 7, 7),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    print('The synases are')
    print(dire.DNA2synapses(DNA))

def DNA2graph_test(x,y):
    DNA=((-1,11,11),(0, 3, 5, 3, 3),(0, 5, 8, 3,3),(0,13,5, x, y),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    print('The graph is')
    dire.imprimir(dire.DNA2graph(DNA))

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

def graph2DNA_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, 7, 7),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    print('The DNA is')
    print(dire.graph2DNA(dire.DNA2graph(DNA)))

def increase_filters_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, 7, 7),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    print('The DNA is')
    print(dire.increase_filters(0,DNA))

def decrease_filters_test(x,y):
    DNA=((-1,x,y),(0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, 3, 3),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    while DNA:
        print('The DNA is')
        DNA=dire.decrease_filters(0,DNA)
        print(DNA)

def add_layer_test(x,y):
    DNA=((-1,1,3,x,y),(0, 3, 5, 3, 3),(0, 5, 8, 3,3),(0,13,5, 5,5 ),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3),(3,3,4),(3,0,2))
    print(dire.add_layer(0,DNA))

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

def fix_fully_conected_test(x,y):
    DNA=((-1,x,y),(0, 3, 5, 3, 3),(0, 5, 5, 3, 3),
            (1, 5, 2), (2,),(3,-1,0),(3,0,1),
            (3,1,2),(3,2,3))
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

#increase_kernel_test(11,11)
decrease_kernel_test(11,11)
#Persistent_synapse_condition_test(11,11)
#add_layer_test(11,11)
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