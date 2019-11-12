import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.P_trees as tr
import utilities.Graphs as gr
from utilities.Abstract_classes.classes.test_stream import TestStream

def test_add_pop():
    stream=TestStream()
    stream.add_node(0)
    stream.charge_nodes()
    stream.add_node(1)
    stream.charge_nodes()
    print('before poping')
    print(stream.findCurrentvalue(0))
    print(stream.findCurrentvalue(1))
    stream.pop()
    print('after poping')
    print(stream.findCurrentvalue(0))
    print(stream.findCurrentvalue(1))

test_add_pop()
