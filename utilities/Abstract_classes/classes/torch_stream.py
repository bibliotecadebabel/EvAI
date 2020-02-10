from utilities.Abstract_classes.AbstractStream import Stream, Charge_log
import children.pytorch.Network as nw
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
            print('the value of p is')
            print(p)
            out_net.Training(data=a[0],
                p=p,
                dt=self.dt,
                labels=a[1])
            out_net.history_loss=[]
            return out_net

    def __init__(self,dataGen,log_size=200,dt=0.01):
        super().__init__(self.Torch_log_creator)
        self.log_size=log_size
        self.dataGen=dataGen
        self.dt=dt

    def charge_node(self,key):
        log=self.key2log(key)
        a=self.dataGen.data
        p=self.log_size
        if log.signal and not(log.log):
            net=log.new_net
            log.old_net=net.clone()
            log.old_net.history_loss=[]
            net.Training(data=a[0],
                p=self.log_size,
                dt=self.dt,
                labels=a[1])
            log.charge(net.history_loss)
            net.history_loss=[]

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


    def sync(self):
        pass



    """def charge_node(self,key,charger=None):
        node=self.Graph.key2node[key]
        node.objects.append(0)"""
