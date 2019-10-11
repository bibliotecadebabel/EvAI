from children.Node_class import Node
import numpy as np
import children.Interfaces as Inter
import children.Net as Net0

N=1000
dt=0.001

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
np.save('w', Net.w_node.Value)

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

Net.pro_node(Net.p_node,x[0],'C')
Inter.Array2image(x[0],'this')
Sc=Net.sc_node.Value
Sn=Net.sn_node.Value
print(Sc)
print(Sn)
p=np.exp(Sc)/(np.exp(Sc)+np.exp(Sn))
print(p)
