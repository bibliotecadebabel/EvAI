import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane
import children.pytorch.NetworkDendrites as nw
import os
import  DNA_graph_functions as program
from Bugs_direction_f import DNA_init

DNA=((-1,1,3,3,32,32),
        (0,3, 15, 3 , 3),
        (0,18, 15, 3,  3),
        (0,33, 50, 32, 32),
        (1, 50,10),
         (2,),
        (3,-1,0),
        (3,0,1),(3,-1,1),
        (3,1,2),(3,0,2),(3,-1,2),
        (3,2,3),
        (3,3,4))

def file2net_test():
    net = nw.Network(DNA, cudaFlag = False)
    file = program.net2file(net)
    print(file)
    net = program.file2net(file,DNA)
    print(net.adn)


def net2file_test():
    net = nw.Network(DNA, cudaFlag = False)
    print({program.net2file(net)})

#DNA2string_test(
file2net_test()
#net2file_test()
