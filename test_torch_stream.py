import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.P_trees as tr
import utilities.Graphs as gr
from utilities.Abstract_classes.classes.test_stream import TestStream
from utilities.Abstract_classes.classes.torch_stream import TorchStream
import children.pytorch.Network as nw
from DAO import GeneratorFromImage
import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim


def create_network(filters,dataGen):
    ks=[filters]
    x = dataGen.size[1]
    y = dataGen.size[2]
    networkADN = ((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
    objects = [x, y, ks[0]]
    network = nw.Network(networkADN
        , objects, cudaFlag=False)
    return network

def create_DNA(filters,dataGen):
    ks=[filters]
    x = dataGen.size[1]
    y = dataGen.size[2]
    networkADN = ((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
    return networkADN

def create_DNAobject(filters,dataGen):
    ks=[filters]
    x = dataGen.size[1]
    y = dataGen.size[2]
    networkADN = ((0, 3, ks[0], x, y), (1, ks[0], 2), (2,))
    objects = [x, y, ks[0]]
    return objects


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
    #print(log.plane)
    a[0]=a[0]+1
    print('After changing a we get')
    #print(log.plane)
    stream.pop()
    stream.pop()
    stream.pop()
    stream.pop()

def test_charge():
    dataGen = GeneratorFromImage.GeneratorFromImage(2, 100, cuda=False)
    dataGen.dataConv2d()
    stream=TorchStream(dataGen,10)
    x = stream.dataGen.size[1]
    y = stream.dataGen.size[2]
    DNA=create_DNA(2,dataGen)
    DNAobject=create_DNAobject(2,dataGen)
    network=create_network(2,dataGen)
    stream.add_node(DNA)
    stream.link_node(DNA,network)
    print('Here')
    log=stream.key2log(DNA)
    stream.charge_nodes()
    i=0
    while i<100:
        print(log.log)
        stream.charge_nodes()
        stream.pop()
        i=i+1


test_charge()