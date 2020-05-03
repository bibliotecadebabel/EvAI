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
    layers=[]
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

def compute_output(g,node):
    key=g.node2key.get(node)
    if len(node.objects)>1:
        pass
    elif key==-1:
        image=node.objects[0]
        node.objects.append([image[1],image[2]])
    else:
        x=0
        y=0
        x_l=node.objects[0][3]
        y_l=node.objects[0][1]
        for parent in node.parents:
            compute_output(g,parent)
            p_out=parent.objects[1]
            x=max(x,p_out[0])
            y=max(y,p_out[1])
        node.objects.append([x-x_l+1,y-y_l+1])

def fix_fully_conected(g):
    full_node=g.key2node.get(len(list(g.key2node.values()))-4)
    compute_output(g,full_node)
    output=full_node.objects[1]
    layer=full_node.objects[0]
    full_node.objects[0]=(layer[0],layer[1],
        layer[2],output[0]+layer[3]-1,output[0]+layer[4]-1)





def graph2DNA(g):
    num_layers=len(list(g.key2node.values()))-1
    node=g.key2node.get(-1)
    DNA=[node.objects[0]]
    for k in range(num_layers):
        node=g.key2node.get(k)
        DNA.append(node.objects[0])
    for k in range(num_layers-2):
        node=g.key2node.get(k)
        parents=node.parents
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
        imprimir(g)
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
    if num_layer>len(DNA2layers(source_DNA))-2:
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
    if num_layer>len(DNA2layers(source_DNA))-2:
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
    def relabler(k):
        if k==-2:
            return num_layer
        elif k<num_layer :
            return k
        else:
            return k+1
    g=DNA2graph(source_DNA)
    total_layers=len(DNA2layers(source_DNA))
    if  num_layer-1>total_layers-3:
        return None
    else:
        node=nd.Node()
        if num_layer==0:
            node.objects.append((0,3,5,3,3))
            g.add_node(-2,node)
            g.add_edges(-1,[-2])
            g.add_edges(-2,[num_layer])
        else:
            o_node=g.key2node.get(num_layer-1)
            o_layer=o_node.objects[0]
            node=nd.Node()
            node.objects.append((0,o_layer[2],5,3,3))
            g.add_node(-2,node)
            g.add_edges(num_layer-1,[-2])
            g.add_edges(-2,[num_layer])
        t_node=g.key2node.get(num_layer)
        t_layer=t_node.objects[0]
        t_layer=layer_chanel(t_layer,5)
        t_node.objects[0]=t_layer
        g.relable(relabler)
        fix_fully_conected(g)
        return graph2DNA(g)

creator=add_layer
directions.update({type:creator})
directions_labels.update({creator:type})

type=(-1,0,0,0)
def remove_layer(num_layer,source_DNA):
    def relabler(k):
        if k<num_layer :
            return k
        else:
            return k-1
    g=DNA2graph(source_DNA)
    total_layers=len(DNA2layers(source_DNA))
    if total_layers<5:
        return None
    else:
        t_node=g.key2node.get(num_layer)
        t_layer=t_node.objects[0]
        for kid in t_node.kids:
            f_layer=kid.objects[0]
            kid.objects[0]=layer_chanel(f_layer,
                -t_layer[2])
        g.remove_node(num_layer)
        if num_layer==0:
            g.add_edges(-1,[1])
            node=g.key2node.get(1)
            node.objects[0]=layer_chanel(node.objects[0],3)
        g.relable(relabler)
        fix_fully_conected(g)
        #imprimir(g)
        return graph2DNA(g)

creator=remove_layer
directions.update({type:creator})
directions_labels.update({creator:type})
