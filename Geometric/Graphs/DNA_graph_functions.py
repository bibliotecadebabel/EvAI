import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import Geometric.TangentPlane as tplane
import children.pytorch.network_dendrites as nw
import os
import time
import statistics
import math

def Alaising(M,m,ep):
    M=10**(-M)
    m=10**(-m)
    return [ m+1/2*(M-m)*(1+pcos(t/e*np.pi))
             for t in range(0,ep)]


def test():
    print('done')

"""def string2DNA(DNA):
    return '('+ ','.join([ DNA2string(layer) if
        type(layer)==tuple else str(layer)
            for layer in DNA]) + ')'"""






def net2file(net,file=None):
    if not(file):
        file=str(time.time()).replace(".", "")
    final_path = os.path.join("temporary_nets", str(file))
    net.save_model(final_path)
    return file

def file2net(file, DNA, cuda_flag = False):
    net= nw.Network(DNA,cuda_flag = False)
    final_path = os.path.join("temporary_nets",file)
    net.load_model(final_path)
    return net




def num_layer2dim_output(DNA,layer):
    pass

def compute_output(g, node):
    key = g.node2key.get(node)
    if len(node.objects) > 1:
        pass
    elif key == -1:
        image = node.objects[0]
        node.objects.append([image[3], image[4]])
    else:
        x = 0
        y = 0
        x_l = node.objects[0][3]
        y_l = node.objects[0][4]
        for parent in node.parents:
            compute_output(g, parent)
            p_out = parent.objects[1]
            x = max(x, p_out[0])
            y = max(y, p_out[1])
        node.objects.append([x-x_l+1, y-y_l+1])

def num_layers(DNA):
    return len([0 for layer in DNA if (layer[0] == 0) or (layer[0] == 4)])

def DNA2num_layers(DNA):
     output=len([0 for layer in DNA if (layer[0] == 0) or (layer[0] == 4)])
     return output

def DNA2synapses(DNA):
    return [layer for layer in DNA if layer[0] == 3]

def DNA2layers(DNA):
    layers=[]
    for layer in DNA:
        if (layer[0]==3):
            break
        else:
            layers.append(layer)
    return layers

def DNA2graph(DNA):
    layers=DNA2layers(DNA)
    synapses=DNA2synapses(DNA)
    g=gr.Graph(True)
    k=-1
    for layer in layers:
        node=nd.Node()
        node.objects.append(layer)
        g.add_node(k,node)
        k=k+1
    for synapse in synapses:
        g.add_edges(synapse[1],[synapse[2]])
    return g

def DNA2size(DNA):
    return sum([
        layer[1]*layer[2]*layer[3]*layer[4] for layer in DNA
        if layer[0]==0])

def add_node(g,DNA):
    node=nd.Node()
    q=qu.Quadrant(DNA)
    p=tplane.tangent_plane()
    node.objects.append(q)
    q.objects.append(p)
    g.add_node(DNA,node)

def node2plane(node):
    q=node.objects[0]
    return q.objects[0]

def node2num_particles(node):
    p=node2plane(node)
    return p.num_particles

def node2energy(node):
    p=node2plane(node)
    return p.energy

def node2direction(node):
    p= node2plane(node)
    return p.direction

def set_num_particles(node,particles):
    p=node2plane(node)
    p.num_particles=particles

def normalize(dataset):

    mean = statistics.mean(dataset) 
    std = statistics.stdev(dataset)

    for i in range(len(dataset)):
        
        norm = (dataset[i] - mean)/std
        dataset[i] = math.exp(norm)
    
    return dataset

def absolute2relative(direction, last_layer, max_layers):

    current_layer = direction[0]

    relative_layer = current_layer - last_layer

    if relative_layer < 0:
        relative_layer = abs(relative_layer)
        relative_layer = relative_layer + max_layers - 1
    
    relative_direction = (relative_layer, direction[1])

    return relative_direction

def relative2absolute(direction, last_layer, max_layers):

    current_relative = direction[0]

    if current_relative >= max_layers:

        current_relative = current_relative - max_layers + 1
        absolute_layer = last_layer - current_relative
    
    else:
        absolute_layer = current_relative + last_layer

    absolute_direction = (absolute_layer, direction[1])

    return absolute_direction

