import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane


def imprimir(g):
    for node in list(g.key2node.values()):
        print('The parent(s) of')
        print(str(g.node2key.get(node))+': '
            +str(node.objects[0]))
        print('are(is):')
        for nodek in node.parents:
            print(str(g.node2key.get(nodek))+': '
                +str(nodek.objects[0]))
        print('and its kids are')
        for nodek in node.kids:
            print(str(g.node2key.get(nodek))+': '
                +str(nodek.objects[0]))

def DNA2layers(DNA):
    layers=[(-1)]
    for layer in DNA:
        if (layer[0]==3):
            break
        else:
            layers.append(layer)
    return layers

def DNA2synapses(DNA):
    synapses=[]
    for layer in DNA:
        if (layer[0]==3):
            synapses.append(layer)
    return synapses


def graph2DNA(g):
    nodes=list(g.key2node.values())
    nodes.pop(0)
    DNA=[]
    for node in nodes:
        DNA.append(node.objects[0])
    for node in nodes:
        parents = node.parents.copy()
        parents.pop(0)
        for parent in parents:
            DNA.append((3,g.node2key.get(parent),
                g.node2key.get(node)))
    return tuple(DNA)

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

def compute_num_layers(DNA):
    k=0
    for layer in DNA:
        if not(layer[0]==3):
            k=k+1
    return k

def layer_filter(layer,N):
    layer_f=list(layer)
    layer_f[2]=layer_f[2]+N
    return tuple(layer_f)

def layer_chanel(layer,N):
    layer_f=list(layer)
    layer_f[1]=layer_f[1]+N
    return tuple(layer_f)

directions={}
directions_labels={}

#linear graph that changes value increases x,y dimension of kernel

type=(0,1,0,0)
def increase_filters(num_layer,source_DNA):
    total_layers=len(DNA2layers(source_DNA))
    if num_layer>total_layers-3:
        return None
    else:
        g=DNA2graph(source_DNA)
        node_t=g.key2node.get(num_layer)
        node_t.objects[0]=layer_filter(node_t.objects[0],
            1)
        for kid in node_t.kids:
            kid.objects[0]=layer_chanel(kid.objects[0],
                1)
        return graph2DNA(g)

creator=increase_filters
directions.update({type:creator})
directions_labels.update({creator:type})

type=(0,-1,0,0)
def decrease_filters(num_layer,source_DNA):
    total_layers=len(DNA2layers(source_DNA))
    if num_layer>total_layers-3:
        return None
    else:
        g=DNA2graph(source_DNA)
        node_t=g.key2node.get(num_layer)
        if node_t.objects[0][2]<3:
            return None
        else:
            node_t.objects[0]=layer_filter(node_t.objects[0],
                -1)
            for kid in node_t.kids:
                kid.objects[0]=layer_chanel(kid.objects[0],
                    -1)
        return graph2DNA(g)

creator=decrease_filters
directions.update({type:creator})
directions_labels.update({creator:type})

def modify_layer_kernel(layer_DNA,num):
    out_DNA=list(layer_DNA)
    if out_DNA[3]+num<2:
        return None
    else:
        out_DNA[3]=out_DNA[3]+num
        out_DNA[4]=out_DNA[4]+num
        return tuple(out_DNA)

type=(0,0,1,1)

def increase_kernel(num_layer,source_DNA):
    if num_layer>len(DNA2layers(source_DNA)))-2:
        return None
    else:
        out_DNA=list(source_DNA)
        layer=list(out_DNA[num_layer])
        layer_f=list(out_DNA[num_layer+1])
        if len(layer_f) == 3:
            return None
        else:
            if modify_layer_kernel(layer,1) and modify_layer_kernel(layer_f,-1):
                out_DNA[num_layer]=modify_layer_kernel(layer,1)
                #out_DNA[num_layer+1]=modify_layer_kernel(layer_f,-1)
                return tuple(out_DNA)
            else:
                return None

creator=increase_kernel
directions.update({type:creator})
directions_labels.update({creator:type})



type=(0,0,-1,-1)
def decrease_kernel(num_layer,source_DNA):
    if num_layer>len(DNA2layers(source_DNA)))-2:
        return None
    else:
        out_DNA=list(source_DNA)
        layer=list(out_DNA[num_layer])
        layer_f=list(out_DNA[num_layer+1])
        if len(layer_f) == 3:
            return None
        else:
            new_layer=modify_layer_kernel(layer,-1)
            if not(new_layer):
                return None
            else:
                out_DNA[num_layer]=new_layer
                #out_DNA[num_layer+1]=modify_layer_kernel(layer_f,1)
                return tuple(out_DNA)


creator=decrease_kernel
directions.update({type:creator})
directions_labels.update({creator:type})

type=(1,0,0,0)
def add_layer(num_layer,source_DNA):
    def relable(k):
        if k==-2:
            return num_layer
        elif:
            k<num_layer
            return k
        else:
            return k+1
    g=graph2DNA(source_DNA)
    total_layers=len(DNA2layers(source_DNA))
    if  num_layer>total_layers-3:
        return None
    else:
        t_node=g.get(num_layer)
        t_layer=t_node.objects[0]
        node=nb.Node()
        node.objects[0]=(0,t_layer[2],5,3,3)
        g.add_node(-2,node)
        f_node=g.get(num_layer+1)
        layer_f=f_node.objects[0]
        f_node.objects[0]=layer_chanel(layer_f,5)

        return tuple(out_DNA)

creator=add_layer
directions.update({type:creator})
directions_labels.update({creator:type})

type=(-1,0,0,0)
def remove_layer(num_layer,source_DNA):
    total_layers=len(source_DNA)
    if total_layers<4 or not(num_layer==0):
        return None
    else:
        out_DNA=list(source_DNA)
        m1_filters=out_DNA[0][2]
        out_DNA.pop(0)
        new_layer_f=list(out_DNA[num_layer])
        new_layer_f[1]=new_layer_f[1]-m1_filters
        out_DNA[num_layer]=tuple(new_layer_f)
        return tuple(out_DNA)

creator=remove_layer
directions.update({type:creator})
directions_labels.update({creator:type})
