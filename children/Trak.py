from Node_class import Node
import numpy as np
import Interfaces as Inter
import Operations as Op
import Net as Net0
import sys
import net2.Network as network

print (sys.argv)

p=2000
dt=0.01
Comp=2
S=100
test=True
k=5

if not(test):
    Target = input("What is the targets name?  ")
    Image = input("What is the image name?  ")
    Net_name = input("What would you like to call the network?")
else:
    Target='folder'
    Image ='btest'
    Net_name='Net_folder'
print('Compressing and Vectorizing input.')
A=Op.Pool(Inter.Image2array(Image),Comp)
x=Op.Pool(Inter.Image2array(Target),Comp)
size=np.shape(x)
print('Sampling Image')
#data=Op.Sample((size[0],size[1]),A,S)
#data.insert(0,x)
data = Op.SampleVer2((size[0],size[1]),A,S, "n")

imageTarget = []
imageTarget.append(x)
imageTarget.append("c")

data.insert(0, imageTarget)

print('Training Net')

#Net=Net0.Network((size[0],size[1]))


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
Inter.trak(Net,A,Net_name+' map')


"""x=np.load('data.npy')
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
