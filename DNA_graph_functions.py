import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane


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
    return len([0 for layer in DNA if layer[0] == 0])

def DNA2num_layers(DNA):
     output=len([0 for layer in DNA if layer[0] == 0])
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

def node2direction(node):
    p= node2plane(node)
    return p.direction

def set_num_particles(node,particles):
    p=node2plane(node)
    p.num_particles=particles
