import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane



directions={}

#linear graph that changes value increases x,y dimension of kernel

type=(0,1,0,0)
def increase_filters(num_layer,source_DNA):
    total_layers=len(source_DNA)
    if num_layer>len(source_DNA)-3:
        return None
    else:
        out_DNA=list(source_DNA)
        layer=list(out_DNA[num_layer])
        layer_f=list(out_DNA[num_layer+1])
        layer[2]=layer[2]+1
        layer_f[1]=layer_f[1]+1
        out_DNA[num_layer]=tuple(layer)
        out_DNA[num_layer+1]=tuple(layer_f)
        return tuple(out_DNA)

creator=increase_filters
directions.update({type:creator})

type=(1,1,0,0)
def increase_filters_first(num_layer,source_DNA):
    total_layers=len(source_DNA)
    if not(num_layer == 0):
        return None
    else:
        return increase_filters(0,source_DNA)

creator=increase_filters_first
directions.update({type:creator})

type=(0,-1,0,0)
def decrease_filters(num_layer,source_DNA):
    total_layers=len(source_DNA)
    if num_layer>len(source_DNA)-3:
        return None
    else:
        out_DNA=list(source_DNA)
        layer=list(out_DNA[num_layer])
        layer_f=list(out_DNA[num_layer+1])
        if layer[2]-1<2:
            return None
        else:
            layer[2]=layer[2]-1
            layer_f[1]=layer_f[1]-1
            out_DNA[num_layer]=tuple(layer)
            out_DNA[num_layer+1]=tuple(layer_f)
            return tuple(out_DNA)

creator=decrease_filters
directions.update({type:creator})

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
    total_layers=len(source_DNA)
    if num_layer>len(source_DNA)-2:
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



type=(0,0,-1,-1)
def decrease_kernel(num_layer,source_DNA):
    total_layers=len(source_DNA)
    if num_layer>len(source_DNA)-2:
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

type=(1,0,0,0)
def add_layer(num_layer,source_DNA):
    total_layers=len(source_DNA)
    if not(num_layer == 0):
        return None
    else:
        out_DNA=list(source_DNA)
        new_layer_f=list(out_DNA[num_layer])
        new_layer_f[1]=new_layer_f[1]+5
        new_layer_o=(0,3,5,3,3)
        out_DNA[num_layer]=tuple(new_layer_f)
        out_DNA.insert(0,tuple(new_layer_o))
        return tuple(out_DNA)

creator=add_layer
directions.update({type:creator})

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
