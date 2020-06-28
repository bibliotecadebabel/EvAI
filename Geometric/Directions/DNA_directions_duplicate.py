import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import Geometric.TangentPlane as tplane


def imprimir(g):
    for node in list(g.key2node.values()):
        print('The parent(s) of')
        print(str(g.node2key.get(node))+': '
            +str(node.objects))
        print('are(is):')
        for nodek in node.parents:
            print(str(g.node2key.get(nodek))+': '
                +str(nodek.objects))
        print('and its kids are')
        for nodek in node.kids:
            print(str(g.node2key.get(nodek))+': '
                +str(nodek.objects))

def DNA2layers(DNA):
    layers=[]
    for layer in DNA:
        if (layer[0]==3):
            break
        else:
            layers.append(layer)
    return layers

def DNA2synapses(DNA):
    return [layer for layer in DNA if layer[0] == 3]

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

def fix_fully_conected(g):
    full_node = g.key2node.get(len(list(g.key2node.values()))-4)
    compute_output(g, full_node)
    output = full_node.objects[1]
    layer = full_node.objects[0]
    full_node.objects[0] = (layer[0],
                            layer[1],
                            layer[2],
                            output[0] + layer[3] - 1,
                            output[0] + layer[4] - 1)

def Persistent_synapse_condition(DNA):
    if DNA:
        g = DNA2graph(DNA)
        full_node = g.key2node.get(len(list(g.key2node.values()))-4)
        compute_output(g, full_node)
        k = 0
        condition = True
        while k < len(list(g.key2node.values()))-4:
            node = g.key2node.get(k)
            output = node.objects[1]
            condition = (condition and (output[0] > 1)
                and (output[1] > 1))
            if not condition:
                return
            k += 1
        return DNA






def graph2DNA(g):
    num_layers=len(list(g.key2node.values()))-1
    node=g.key2node.get(-1)
    DNA=[node.objects[0]]
    for k in range(num_layers):
        node=g.key2node.get(k)
        DNA.append(node.objects[0])
    for k in range(num_layers):
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
    if num_layer>total_layers-4:
        return None
    else:
        g=DNA2graph(source_DNA)
        #imprimir(g)
        node_t=g.key2node.get(num_layer)
        increase_amount = node_t.objects[0][2]
        node_t.objects[0]=layer_filter(node_t.objects[0],
            increase_amount)
        for kid in node_t.kids:
            kid.objects[0]=layer_chanel(kid.objects[0],
                increase_amount)
        return graph2DNA(g)

creator=increase_filters
directions.update({type:creator})
directions_labels.update({creator:type})

type=(0,-1,0,0)
def decrease_filters(num_layer,source_DNA):
    total_layers=len(DNA2layers(source_DNA))
    if num_layer>total_layers-4:
        return None
    else:
        g=DNA2graph(source_DNA)
        node_t=g.key2node.get(num_layer)
        if node_t.objects[0][2]<3:
            return None
        else:
            decrease_amount = node_t.objects[0][2] // 2
            node_t.objects[0]=layer_filter(node_t.objects[0],
                -decrease_amount)
            for kid in node_t.kids:
                kid.objects[0]=layer_chanel(kid.objects[0],
                    -decrease_amount)
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
    if num_layer>len(DNA2layers(source_DNA))-4:
        return None
    else:
        num_layer=num_layer+1
        out_DNA=list(source_DNA)
        layer=list(out_DNA[num_layer])
        layer_f=list(out_DNA[num_layer+1])
        if len(layer_f) == 3:
            return None
        else:
            if modify_layer_kernel(layer,1):
                out_DNA[num_layer]=modify_layer_kernel(layer,1)
                #out_DNA[num_layer+1]=modify_layer_kernel(layer_f,-1)
                DNA=tuple(out_DNA)
                g=DNA2graph(DNA)
                fix_fully_conected(g)
                return Persistent_synapse_condition(graph2DNA(g))
            else:
                return None

creator=increase_kernel
directions.update({type:creator})
directions_labels.update({creator:type})



type=(0,0,-1,-1)
def decrease_kernel(num_layer,source_DNA):
    if num_layer>len(DNA2layers(source_DNA))-4:
        return None
    else:
        num_layer=num_layer+1
        out_DNA=list(source_DNA)
        layer=list(out_DNA[num_layer])
        layer_f=list(out_DNA[num_layer+1])
        if len(layer_f) == 3:
            return None
        else:
            if modify_layer_kernel(layer,-1):
                out_DNA[num_layer]=modify_layer_kernel(layer,-1)
                #out_DNA[num_layer+1]=modify_layer_kernel(layer_f,-1)
                DNA=tuple(out_DNA)
                g=DNA2graph(DNA)
                fix_fully_conected(g)
                return Persistent_synapse_condition(graph2DNA(g))
            else:
                return None


creator=decrease_kernel
directions.update({type:creator})
directions_labels.update({creator:type})

type=(1,0,0,0)
def add_layer(num_layer,source_DNA):
    if num_layer>len(DNA2layers(source_DNA))-3:
        return None
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
        return Persistent_synapse_condition(graph2DNA(g))

creator=add_layer
directions.update({type:creator})
directions_labels.update({creator:type})

type=(-1,0,0,0)
def remove_layer(num_layer,source_DNA):
    if num_layer>len(DNA2layers(source_DNA))-4:
        return None
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
        node=g.key2node.get(num_layer+1)
        node_o=g.key2node.get(num_layer-1)
        if not(node_o in node.parents):
            g.add_edges(num_layer-1,[num_layer+1])
            node.objects[0]=layer_chanel(node.objects[0],
                node_o.objects[0][2])
        #print('The new graph is:')
        #imprimir(g)
        g.relable(relabler)
        fix_fully_conected(g)
        return Persistent_synapse_condition(graph2DNA(g))

creator=remove_layer
directions.update({type:creator})
directions_labels.update({creator:type})

type=(0,0,1)
def spread_dendrites(num_layer,source_DNA):
    total_layers=len(DNA2layers(source_DNA))
    num_layer=num_layer-1
    if num_layer>total_layers-5:
        return None
    g=DNA2graph(source_DNA)
    node=g.key2node.get(num_layer)
    dendrites=node.kids.copy()
    dendrites.remove(g.key2node.get(num_layer+1))
    landscape=[g.node2key.get(node_k)-num_layer-1
        for node_k in dendrites]
    old_index=select_old_index2spread(num_layer,
        landscape,total_layers-num_layer-5)
    if old_index:
#        print(f'The idex to remove is {old_index}')
        g.remove_edge(num_layer,old_index)
        t_node=g.key2node.get(old_index)
        t_layer=t_node.objects[0]
        t_node.objects[0]=layer_chanel(t_layer,-node.objects[0][2])
        dendrites.remove(g.key2node.get(old_index))
        landscape=[g.node2key.get(node_k)-num_layer-1
            for node_k in dendrites]
    new_index=select_new_index2spread(num_layer,
        landscape,total_layers-num_layer-5)
#    print(f'The idex to add is {new_index}')
    if new_index and not(new_index==old_index) and (
        not(new_index==num_layer or new_index==num_layer+1)):
        g.add_edges(num_layer,[new_index])
        t_node=g.key2node.get(new_index)
        t_layer=t_node.objects[0]
        t_node.objects[0]=layer_chanel(t_layer,node.objects[0][2])
#        print('The new graph is')
        fix_fully_conected(g)
        #imprimir(g)
        return Persistent_synapse_condition(graph2DNA(g))
    else:
        return None


creator=spread_dendrites
directions.update({type:creator})
directions_labels.update({creator:type})

def select_old_index2spread(num_layer,landscape,size):
    if len(landscape)<5:
        return None
    else:
        k=0
        landscape_dif=[]
        for dendrite in landscape:
            if k==0:
                landscape_dif.append(dendrite)
            else:
                landscape_dif.append(abs(dendrite-landscape[k-1]))
            k=k+1
        #print('The dif_landscape is')
        #print(landscape_dif)
        index = landscape_dif.index(min(landscape_dif))
        return num_layer+landscape[index]+1

def select_new_index2spread(num_layer,landscape,size):
    #print(f'The landscape is {landscape}')
    #print(f'The size is {size}')
    if len(landscape)<1:
        if size>0:
            return num_layer+2
        else:
            return None
    elif len(landscape)==1:
        if size==1:
            return None
        l_o=landscape[0]
        new_index=min(2*l_o+1,size)
        if l_o==new_index:
            #print(f'The new index is {new_index-1}')
            return num_layer+new_index-1+1
        else:
            #print(f'The new index is {new_index}')
            return num_layer+1+new_index
    else:
        k=0
        landscape_dif=[]
        for dendrite in landscape:
            if k==0:
                landscape_dif.append(dendrite)
            else:
                landscape_dif.append(abs(dendrite-landscape[k-1]))
            k=k+1
        spread=max(landscape_dif)
        new_index=max(landscape)+spread
        if new_index<size:
            return num_layer+new_index+1
        elif (new_index==size) and not(size in landscape):
            return num_layer+size+1
        else:
            base_index=landscape_dif.index(spread)
            new_index=landscape[base_index]+spread//2+1
            if not(new_index in landscape) and new_index<size:
                return num_layer+new_index+1
            else:
                return None

type=(0,0,-1)
def retract_dendrites(num_layer,source_DNA):
    total_layers=len(DNA2layers(source_DNA))
    num_layer=num_layer-1
    if num_layer>total_layers-5:
        return None
    g=DNA2graph(source_DNA)
    node=g.key2node.get(num_layer)
    dendrites=node.kids.copy()
    dendrites.remove(g.key2node.get(num_layer+1))
    landscape=[g.node2key.get(node_k)-num_layer-1
        for node_k in dendrites]
    old_index=select_old_index2retract(num_layer,
        landscape,total_layers-num_layer-5)
    if old_index:
        print(f'The target layer is {num_layer} and the index is {old_index}')
        g.remove_edge(num_layer,old_index)
        t_node=g.key2node.get(old_index)
        t_layer=t_node.objects[0]
        t_node.objects[0]=layer_chanel(t_layer,-node.objects[0][2])
        dendrites.remove(g.key2node.get(old_index))
        landscape=[g.node2key.get(node_k)-num_layer-1
            for node_k in dendrites]
    new_index=select_new_index2retract(num_layer,
        landscape,total_layers-num_layer-5,old_index)
#    print(f'The idex to add is {new_index}')
    if new_index and not(new_index==old_index) and (
        not(new_index==num_layer or new_index==num_layer+1)):
        g.add_edges(num_layer,[new_index])
        t_node=g.key2node.get(new_index)
        t_layer=t_node.objects[0]
        t_node.objects[0]=layer_chanel(t_layer,node.objects[0][2])
#        print('The new graph is')
        fix_fully_conected(g)
        #imprimir(g)
        return Persistent_synapse_condition(graph2DNA(g))
    else:
        if not(old_index):
            return None
        else:
            fix_fully_conected(g)
            return Persistent_synapse_condition(graph2DNA(g))


creator=retract_dendrites
directions.update({type:creator})
directions_labels.update({creator:type})

def select_old_index2retract(num_layer,landscape,size):
    if len(landscape)<1:
        return None
    else:
        return num_layer+max(landscape)+1

def select_new_index2retract(num_layer,landscape,size,old_index=None):
    if len(landscape)<1:
        if old_index:
            contract=old_index-num_layer+1
            if contract//2==0:
                return None
            else:
                return num_layer+contract//2
        else:
            return None
    else:
        k=0
        landscape_dif=[]
        for dendrite in landscape:
            if k==0:
                landscape_dif.append(dendrite)
            else:
                landscape_dif.append(abs(dendrite-landscape[k-1]))
            k=k+1
        spread=max(landscape_dif)
        base_index=landscape_dif.index(spread)-1
        if base_index==-1:
            new_index=spread//2
        else:
            new_index=landscape[base_index]+spread//2
        if not(new_index in landscape):
            return num_layer+new_index+1
        else:
            return None
