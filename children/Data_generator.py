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
        self.S=1000
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


"""
networkParameters = np.full((3), (size[0], size[1], k))
Net = network.Network(networkParameters)
l=['C']
i=0
while i<S:
    l.append('N')
    i=i+1
Net.Training(data=data, dt=dt, p=p)
print("finish training")
np.save(Net_name+' w-node',Net.nodes[0].objects[0].value)

print('Scaning Image')
Inter.trak(Net,A,Net_name+' map')"""

"""
x=np.load('data.npy')
Shap=np.shape(x)
l=['C']
i=0
while i<Shap[0]-1:
    l.append('N')
    i=i+1
size=(np.shape(x[0]))
Net=Net0.Network((size[0],size[1]))
Net.train(x,l,N,dt)
np.save('w', Net.w_node.Value)"""

"""x=np.zeros((2,2,3),dtype=np.uint8)
x[0,0,1]=300
size=np.shape(x)
Net=Net0.Network((size[0],size[1]))
print(Net.sc_node.Value)
Net.pro_node(Net.p_node,x,'C')
#Net.pro_node(Net.sn_node,x,'C')
print(Net.sc_node.Value)
print(Net.sn_node.Value)
print(Net.p_node.Value)
#Net.bak([x],['C'])
#print(Net.w_node.Der[0])
Net.train([x],'C',N,dt)"""

"""Net.pro_node(Net.p_node,x[0],'C')
Inter.Array2image(x[0],'this')
Sc=Net.sc_node.Value
Sn=Net.sn_node.Value
print(Sc)
print(Sn)
p=np.exp(Sc)/(np.exp(Sc)+np.exp(Sn))
print(p)"""
