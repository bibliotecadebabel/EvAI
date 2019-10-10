from Node_class import Node
import numpy as np
import Interfaces as Inter

class Network():
    def __init__(self,size):
        self.w_node=Node(np.zeros((2,size[0], size[1], 3), dtype=np.float64),
                        np.zeros((2,size[0], size[1], 3), dtype=np.float64),
                        'W')
        self.r_node=Node(0,0,'R')
        self.sc_node=Node(0,0,'Sc')
        self.sn_node=Node(0,0,'Sn')
        self.p_node=Node(0,0,'P')
        self.l_node=Node(0,1,'L')

        self.w_node.kids.append(self.sc_node)
        self.w_node.kids.append(self.sn_node)
        self.w_node.kids.append(self.r_node)
        self.sc_node.kids.append(self.p_node)
        self.sc_node.parents.append(self.w_node)
        self.sn_node.kids.append(self.p_node)
        self.sn_node.parents.append(self.w_node)
        self.p_node.parents.append(self.sc_node)
        self.p_node.parents.append(self.sn_node)
        self.r_node.parents.append(self.w_node)
        a=Inter.Random_array(size[0],size[1])
        b=Inter.Random_array(size[0],size[1])
        self.w_node.Value[0]=a
        self.w_node.Value[1]=b



    def pre(self,x):
        self.pro_node(self.sc_node,x)
        self.pro_node(self.sn_node,x)
        Sc=self.sc_node.Value
        Sn=self.sn_node.Value
        return np.exp(Sc)/(np.exp(Sc)+np.exp(Sn))

    def train(self,x,l,p,dt):
        i=0
        j=0
        while j<p:
            self.bak(x,l)
            a=self.w_node.Value
            self.w_node.Value=a-dt*self.w_node.Der
            self.w_node.Der=a*0
            #print(self.w_node.Value)
            j=self.pre(x[0])
            if i % 10==0:
                print(j)
            i=i+1

    def pro(self, x, label):
        L=0
        for i in range(len(label)):
            self.pro_node(self.p_node, x[i], label[i])
            p_i=self.p_node.Value
            if i==0:
                L=len(label)/4*p_i
            else:
                L=p_i+L
        self.pro_node(self.r_node, x[i], label[i])
        self.l_node.Value=self.r_node.Value+L/(len(label)*5/4)

    def bak(self, x, label):
        for i in range(len(label)):
            self.pro_node(self.p_node, x[i], label[i])
            self.bak_node(self.w_node, x[i], label[i])
            if i==0:
                self.w_node.Der=self.w_node.Der*len(label)/4
        self.w_node.Der=self.w_node.Value/(np.size(x)*255)+self.w_node.Der/(len(label)*5/4)


    def pro_node(self, node, x=None, label=None):
        if node.type=='W':
            pass
        if node.type=='R':
            a=node.parents[0].Value
            b=a*a
            node.Value=b.sum()*1/(np.size(x)*255)
        if node.type=='Sc':
            a=self.w_node.Value[0]
            b=a*x
            node.Value=b.sum()*1/(np.size(x)*255)
        if node.type=='Sn':
            a=node.parents[0].Value[1]
            b=a*x
            node.Value=b.sum()*1/(np.size(x)*255)
        if node.type=='P':
            if label is not None:
                for parent in node.parents:
                    self.pro_node(parent,x,label)
                Sc=node.parents[0].Value
                Sn=node.parents[1].Value
                if label=='C':
                    node.Value=-np.log(np.exp(Sc)/(np.exp(Sc)+np.exp(Sn)))
                if label=='N':
                    node.Value=-np.log(np.exp(Sn)/(np.exp(Sc)+np.exp(Sn)))
    def bak_node(self, node, x=None, label=None):
        if node.type=='W':
            for kid in node.kids:
                self.bak_node(kid, x, label)
            node.Der[0]=node.Der[0]+x*node.kids[0].Der
            node.Der[1]=node.Der[1]+x*node.kids[1].Der
        if node.type=='R':
            pass
        if node.type=='Sc':
            Sc=self.sc_node.Value
            Sn=self.sn_node.Value
            if label=='C':
                Sk=node.Value
                a=np.exp(Sk)/(np.exp(Sc)+np.exp(Sn))
                node.Der=-(a-a*a)/a
            else:
                Sk=self.sn_node.Value
                a=np.exp(Sk)/(np.exp(Sc)+np.exp(Sn))
                node.Der=-(-a*a*(np.exp(Sc)/np.exp(Sk)))/a
        if node.type=='Sn':
            Sc=self.sc_node.Value
            Sn=self.sn_node.Value
            if label=='N':
                Sk=node.Value
                a=np.exp(Sk)/(np.exp(Sc)+np.exp(Sn))
                node.Der=-(a-a*a)/a
            else:
                Sk=self.sc_node.Value
                a=np.exp(Sk)/(np.exp(Sc)+np.exp(Sn))
                node.Der=-(-a*a*(np.exp(Sn)/np.exp(Sk)))/a
        if node.type=='P':
            node.Der=1






#x=Inter.Image2array('Target')
#size=(np.shape(x))
#Net=Network((size[0],size[1]))
#a=Inter.Random_array(size[0],size[1])
#b=Inter.Random_array(size[0],size[1])
#Net.w_node.Value[0]=a
#Net.w_node.Value[1]=b
#Net.pro_node(Net.sc_node,x,'C')
#Net.pro_node(Net.sn_node,x,'C')
#Net.pro_node(Net.p_node,x,'C')
#Net.bak([x],['C'])
#print(Net.w_node.Der[0])
#print(Net.sc_node.Value)
#print(Net.sn_node.Value)
#i=0
#print(Net.w_node.Value[0])
#Inter.Array2image(Net.w_node.Value[0],'test')
#np.save('test3.npy', Net.w_node.Der[0])
