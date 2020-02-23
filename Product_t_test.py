import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Safe as sf
import utilities.P_trees as tr
import numpy as np
import TangentPlane as tplane
import utilities.Graphs as gr
import math
import V_graphics as cd
import Transfer.Transfer as tran
import children.Data_generator as dgen
import children.Interfaces as Inter
import children.Operations as Op
import children.net2.Network as nw
from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
from DNA_Phase_space import DNA_Phase_space
from Dynamic_DNA import Dynamic_DNA
from utilities.Abstract_classes.classes.torch_stream import TorchStream
import children.pytorch.Network as nw
import time
import Product_t as Pt

def initialize():
    status=Pt.Status()
    Pt.initialize_parameters(status)
    print('initialize parameters done')

def create_objects():
    status=Pt.Status()
    Pt.initialize_parameters(status)
    Pt.create_objects(status)
    print('create objects Done done')

def update():
    status=Pt.Status()
    Pt.initialize_parameters(status)
    Pt.create_objects(status)
    Pt.update(status)
    print('update done')

#initialize()
#create_objects()
update()
