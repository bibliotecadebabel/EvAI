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
    if not(DNA):
        return False
    else:
        if all([layer[2]<max for layer in DNA if layer[0] == 0]):
            return DNA
        else:
            return False
