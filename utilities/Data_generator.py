#from Node_class import Node
import numpy as np
import children.Interfaces as Inter
import children.Operations as Op
import sys
import children.net2.Network as network

print (sys.argv)

class Data_gen():
    def __init__(self):
        self.Comp = 2
        self.S=100
        self.Image='btest'
        self.Target='folder'
        self.Data=[]
        self.size=None
    def gen_data(self):
        print('Compressing and Vectorizing input.')
        A=Op.Pool(Inter.Image2array(self.Image),self.Comp)
        x=Op.Pool(Inter.Image2array(self.Target),self.Comp)
        size=np.shape(x)
        print('Sampling Image')
        self.Data = Op.SampleVer2((size[0],size[1]),A,self.S, "n")
        imageTarget = []
        imageTarget.append(x)
        imageTarget.append("c")
        self.Data.insert(0, imageTarget)
        self.size=size
        self.A = A
        self.x = x


