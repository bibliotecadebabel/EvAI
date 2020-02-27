import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr


def max_layer(DNA,max):
    if (DNA== None) or  (len(DNA)-2>max) :
        return False
    else:
        return DNA
