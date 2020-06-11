from utilities.Abstract_classes.AbstractStream import Stream, Charge_log
import children.pytorch.NetworkDendrites as nw
from DAO import GeneratorFromImage
import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim
from DNA_graph_functions import DNA2size
import time



class TorchStream(Stream):
    def Alai2creator(self,Alai):
        class Torch_log_creator(Charge_log):
            def __init__(self,a=0):
                super().__init__()
                self.new_net=None
                self.log_size=None
                self.dataGen=None
                self.Alai=Alai
                self.dt=0
            def get_net(self):
                out_net=self.new_net.clone()
                out_net.history_loss=[]
                return out_net
        return Torch_log_creator
    def __init__(self,dataGen,log_size=200,dt=0.001,min_size=5,
        Alai=None,status=None):
        self.Torch_log_creator=self.Alai2creator(Alai)
        super().__init__(self.Torch_log_creator)
        self.log_size=log_size
        self.dataGen=dataGen
        self.cuda=self.dataGen.cuda
        self.dt=dt
        self.min_size=min_size
        self.Alai=Alai
        self.flags=True
        if status:
            self.version=status.version
            self.settings=status.settings
            self.status
            if status.cuda:
                import torch
        else:
            self.version='clone'
            self.status=None

    def flags_print(self,text):
        if self.flags==True:
            print(text)
            time.sleep(.5)

    def key2average(self,key):
        log=self.key2log(key)
        history=log.log.copy()
        return sum(history[:self.min_size])/self.min_size

    def key2len_hist(self,key):
        log=self.key2log(key)
        history=log.log.copy()
        return len(history)

    def charge_node(self,key):
        log=self.key2log(key)
        a=self.dataGen.data
        p=self.log_size
        if log.signal and not(log.log):
#            print('The net')
#            print(key)
#            print('is charging')
            net=log.new_net
            Alai=self.Alai
            if not(self.Alai):
                self.flags_print(f'Training no Alaising : dt={self.dt}')
                net.Training(data=self.dataGen,
                    p=self.log_size,
                    dt=self.dt,full_database=True)
            else:
                dt=Alai.get_increments(self.log_size)
                self.flags_print(f'The training range is dt_max : {max(dt)}, dt_min :{min(dt)} ')
                if self.status:
                    if self.status.cuda:
                        torch.cuda.empty_cache()
                net.iterTraining(dataGenerator=self.dataGen,
                    dt_array=Alai.get_increments(self.log_size))
            log.charge(net.history_loss)
            net.history_loss=[]
        elif log.signal and (len(log.log) < self.min_size+2):
#            print('The net')
#            print(key)
#            print('is charging')
            net=log.get_net()
            Alai=self.Alai
            if not(self.Alai):
                self.flags_print(f'Training no Alaising : dt={self.dt}')
                net.Training(data=self.dataGen,
                    p=self.log_size-self.min_size,
                    dt=self.dt,full_database=True)
            else:
                dt=Alai.get_increments(self.log_size)
                self.flags_print(f'The training range is dt_max : {max(dt)}, dt_min :{min(dt)} ')
                if self.status:
                    if self.status.cuda:
                        torch.cuda.empty_cache()
                net.iterTraining(dataGenerator=self.dataGen,
                    dt_array=dt)
#            net.Training(data=self.dataGen,
#                p=self.log_size-5,
#                dt=self.dt,full_database=True)
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
            log.new_net=net


    def add_net(self,key):
        node=self.key2node(key)
        if not(node):
            self.add_node(key)
            settings=self.settings
            network = nw.Network(key,cudaFlag=settings.cuda,
             momentum=settings.momentum,
             weight_decay=settings.weight_decay,
             enable_activation=settings.enable_activation,
             enable_track_stats=settings.enable_track_stats,
             dropout_value=settings.dropout_value,
             dropout_function=settings.dropout_function,
             enable_last_activation=settings.enable_last_activation,
             version=settings.version, eps_batchnorm=settings.eps_batchorm
             )
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
            if log.signal:
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

    def clear(self):
        Graph = self.Graph
        graph_dict = Graph.key2node
        keys2erase=[]
        nodes2erase=[]
        for key, node in graph_dict.items():
            log = node.get_object()
            log = self.key2log(key)
            if log:
                if log.signal == False:
                    if log.new_net:
                        del log.new_net
                    keys2erase.append(key)
                    nodes2erase.append(node)
                else:
                    log.new_net.history_loss=[]
        for k in range(len(keys2erase)):
            Graph.key2node.pop(keys2erase[k])
            Graph.node2key.pop(nodes2erase[k])
        del keys2erase
        del nodes2erase




    """def charge_node(self,key,charger=None):
        node=self.Graph.key2node[key]
        node.objects.append(0)"""
