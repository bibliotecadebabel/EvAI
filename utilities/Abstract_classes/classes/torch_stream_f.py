from utilities.Abstract_classes.AbstractStream import Stream, Charge_log
import children.pytorch.NetworkDendrites as nw
from DAO import GeneratorFromImage
import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim

class TorchStream(Stream):
    class Torch_log_creator(Charge_log):
        def __init__(self):
            super().__init__()
            self.old_net=None
            self.new_net=None
            self.log_size=None
            self.dataGen=None
            self.dt=0
        def get_net(self):
            p=self.log_size-len(self.log)
            out_net=self.old_net.clone()
            a=self.dataGen.data
            out_net.Training(data=self.dataGen,
                p=p,
                dt=self.dt)
            out_net.history_loss=[]
            return out_net

    def __init__(self,dataGen,log_size=200,dt=0.001):
        super().__init__(self.Torch_log_creator)
        self.log_size=log_size
        self.dataGen=dataGen
        self.dt=dt

    def charge_node(self,key):
        log=self.key2log(key)
        a=self.dataGen.data
        p=self.log_size
        if log.signal and not(log.log):
#            print('The net')
#            print(key)
#            print('is charging')
            net=log.new_net
            log.old_net=net.clone()
            log.old_net.history_loss=[]
            net.Training(data=self.dataGen,
                p=self.log_size,
                dt=self.dt)
            log.charge(net.history_loss)
            net.history_loss=[]
        elif log.signal and (len(log.log) <5):
#            print('The net')
#            print(key)
#            print('is charging')
            net=log.get_net()
            log.old_net=net.clone()
            log.old_net.history_loss=[]
            net.Training(data=self.dataGen,
                p=self.log_size-5,
                dt=self.dt)
            log.charge(net.history_loss)
            net.history_loss=[]
#        else:
#            print('The net')
#            print(key)
#            print('is not charging')
#            print('The size of its log is')
#            print(len(log.log))

    def key2signal_on(self,key):
        log=self.key2log(key)
        if log:
            log.signal=True
        pass

    def key2signal_off(self,key):
        log=stream.key2log(key)
        if log:
            log.signal=False
        pass

    def link_node(self,key,net=None):
        log=self.key2log(key)
        log.signal=True
        log.dataGen=self.dataGen
        log.log_size=self.log_size
        log.dt=self.dt
        if net is not None:
            log.old_net=net
            log.new_net=net


    def add_net(self,key):
        node=self.key2node(key)
        if not(node):
            self.add_node(key)
            network = nw.Network(key,
                                 cudaFlag=False)
            self.link_node(key,network)
            self.charge_node(key)
            print('added net')

    def get_net(self,key):
        log=self.key2log(key)
        if log:
            return log.get_net()
        else:
            return log

    def imprimir(self):
        Graph = self.Graph
        graph_dict = Graph.key2node
        k = 0
        for key, node in graph_dict.items():
            log = node.get_object()
            print('The energy of {} is {}'.format(key, log.Currentvalue()))
            k += 1
    def print_signal(self):
        Graph = self.Graph
        graph_dict = Graph.key2node
        k = 0
        for key, node in graph_dict.items():
            log = node.get_object()
            print('The signal of {} is {}'.format(key, log.signal))
            k += 1

    def sync(self):
        pass

    def signals_off(self):
        Graph = self.Graph
        graph_dict = Graph.key2node
        for key, node in graph_dict.items():
            log = node.get_object()
            log = self.key2log(key)
            if log:
                log.signal = False



    """def charge_node(self,key,charger=None):
        node=self.Graph.key2node[key]
        node.objects.append(0)"""
