import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr


def max_layer(DNA,max):
    if not(DNA):
        return False
    else:
        num_layer=len([0 for layer in DNA if layer[0] == 0])
        if (num_layer<max+1):
            return DNA
        else:
            return False

def max_filter(DNA,max):
    u=True
    if DNA:
        for i in range(len(DNA)-1):
            u=u and DNA[i][1]<max
            if not(u):
                break
        if not(u):
            return False
        else:
            return DNA
    else:
        return DNA
