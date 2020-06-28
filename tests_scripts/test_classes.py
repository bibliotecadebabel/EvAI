import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.P_trees as tr
import utilities.Graphs as gr
from utilities.Abstract_classes.classes.test_stream import TestStream
from utilities.Abstract_classes.classes.torch_stream import TorchStream

def test_stream():
    stream=TestStream()
    stream.add_node(0)
    print('The current value of 0 is')
    print(stream.findCurrentvalue(0))
    stream.charge_nodes()
    print('The current value of 0, after charging is')
    print(stream.findCurrentvalue(0))
    stream.pop()
    print('The current value of 0, after poping is')
    print(stream.findCurrentvalue(0))

def torch_stream_add():
    log_size=2
    stream=TorchStream(4)
    stream.add_node(0)
    print('The current value of 0 is')
    print(stream.findCurrentvalue(0))
    stream.charge_nodes()
    print('The current value of 0, after charging is')
    print(stream.findCurrentvalue(0))
    stream.pop()
    print('The current value of 0, after poping is')
    print(stream.findCurrentvalue(0))
    print('After linking 0 we get')
    a=[15]
    stream.link_node(0,a)
    log=stream.key2log(0)
    print('Before changing a we get')
    print(log.plane)
    a[0]=a[0]+1
    print('After changing a we get')
    print(log.plane)
    stream.pop()
    stream.pop()
    stream.pop()
    stream.pop()

def torch_stream_charge():
    log_size=2
    stream=TorchStream(4)
    stream.add_node(0)
    print('The current value of 0 is')
    print(stream.findCurrentvalue(0))
    stream.charge_nodes()
    print('The current value of 0, after charging is')
    print(stream.findCurrentvalue(0))
    stream.pop()
    print('The current value of 0, after poping is')
    print(stream.findCurrentvalue(0))
    print('After linking 0 we get')
    a=[15]
    stream.link_node(0,a)
    log=stream.key2log(0)
    print('Before changing a we get')
    print(log.plane)
    a[0]=a[0]+1
    print('After changing a we get')
    print(log.plane)
    stream.pop()
    stream.pop()
    stream.pop()
    stream.pop()


torch_stream_add()
