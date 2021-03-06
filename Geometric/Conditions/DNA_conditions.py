import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import Geometric.Graphs.DNA_graph_functions as Fun
import Geometric.Graphs.DNA_graph_functions as Fun


def max_layer(DNA,max):
    if not(DNA):
        return False
    else:
        num_layer=len([0 for layer in DNA if layer[0] == 0])
        if (num_layer<max+1):
            return DNA
        else:
            return False
"""
def max_filter(DNA,max):
    u=True
    if not(DNA):
        return False
    else:
        if all([layer[2]<max for layer in DNA if layer[0] == 0]):
            return DNA
        else:
            return False
"""

def max_filter(DNA,max):
    u=True
    if not(DNA):
        return False
    else:
        num_layer=len([0 for layer in DNA if layer[0] == 0])
        if all([DNA[k][2]<max for k in range(num_layer)]):
            return DNA
        else:
            return False

def min_filter(DNA,max):
    u=True
    if not(DNA):
        return False
    else:
        num_layer=len([0 for layer in DNA if layer[0] == 0])
        if all([DNA[k+1][2]>max for k in range(num_layer-1)]):
            return DNA
        else:
            return False

def max_filter_dense(DNA,max):
    u=True
    if not(DNA):
        return False
    else:
        num_layer=len([0 for layer in DNA if layer[0] == 0])
        if DNA[num_layer][2]<max:
            return DNA
        else:
            return False

def max_kernel_dense(DNA,max):
    u=True
    if not(DNA):
        return False
    else:
        num_layer=len([0 for layer in DNA if layer[0] == 0])
        if DNA[num_layer][3]<max+1 and DNA[num_layer][4]<max+1:
            return DNA
        else:
            return False

def max_pool_layer(DNA,max):
    u=True
    if not(DNA):
        return False
    else:
        num_pool_layer=len([0 for layer in DNA if len(layer) == 6])
        if num_pool_layer<max+1:
            return DNA
        else:
            return False


def restrict_conections(DNA,max=None):
    return no_con_last_layer(
        no_con_image(DNA))

def no_con_last_layer(DNA,max=None):
    if DNA:
        g=Fun.DNA2graph(DNA)
        node=g.key2node.get(Fun.num_layers(DNA)-1)
        if len(node.parents)==1:
            return DNA
        else:
            return False

def no_con_image(DNA,max=None):
    if DNA:
        g=Fun.DNA2graph(DNA)
        node=g.key2node.get(-1)
        if len(node.kids)==1:
            return DNA
        else:
            return False

def max_parents(DNA,max):
    if DNA:
        g=Fun.DNA2graph(DNA)
        if all([len(node.parents)<max+1
            for node in list(g.key2node.values()) ]):
            return DNA
        else:
            return False
def test_function(DNA,max):
    return max

def dict2condition(DNA,dict):
    for key in dict.keys():
        DNA=key(DNA,dict.get(key))
    return DNA

def con_image(DNA,max=None):
    if DNA:
        g=Fun.DNA2graph(DNA)
        condition= all([node_con_image(g,node) for node
            in list(g.key2node.values())])
        if condition:
            return DNA
        else:
            return None

def node_con_image(g,node):
    if g.node2key.get(node) == -1:
        return True
    elif len(node.parents)==0:
        return False
    else:
        output = any([node_con_image(g,parent) for parent in node.parents])
        return output

def is_convex(DNA,index):
    g=Fun.DNA2graph(DNA)
    node=g.key2node.get(index)
    output = node.objects[0][1]==max([node.objects[0][2] for node
        in node.parents])
    return output
