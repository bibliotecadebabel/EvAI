from children.Node_class import Node
import numpy as np
import children.Interfaces as Inter
import children.Net as Net0

N=1000
dt=0.001

a=np.load('w.npy')
data=np.load('data.npy')
print(np.shape(data))

"""a=np.load('w.npy')
Inter.Array2image(a[0],'w0')
size=np.shape(a)
Net=Net0.Network((size[0],size[1]))
Net.w_node.Value=a
data=np.load('data.npy')
print(Net.pre(data[0]))
print(Net.pre(data[1000]))
Inter.traking(Net,'screen','map')"""
