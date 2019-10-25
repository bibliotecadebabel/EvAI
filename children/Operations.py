import numpy as np
import children.Interfaces as Inter
import random

def Pool(A,N=None):
    size=np.shape(A)
    i=0
    j=0
    sizeb=np.array([np.floor(size[0]/2),np.floor(size[1]/2)])
    sizeb.astype(int)
    A.astype(int)
    a=np.zeros((int(sizeb[0]),int(sizeb[1]), 3), dtype=np.uint8)
    a=a
    while i<size[0]-1:
        j=0
        while  j<size[1]-1:
            x=A[i,j].astype(int)+A[i+1,j].astype(int)+A[i,j+1].astype(int)+A[i+1,j+1].astype(int)
            y=np.array(x/4,dtype=np.uint8)
            a[int(i/2),int(j/2)]=y
            j=j+2
        i=i+2
    if N==None:
        N=1
    if not(N==1):
        print(N)
        a=Pool(a,N-1)
    return a

def Conv(A,B):
    size=np.shape(B)
    sizeb=np.shape(A)
    i=0
    j=0
    a=np.zeros((sizeb[0]-size[0]+1,sizeb[1]- size[1]+1), dtype=np.float64)
    while i<sizeb[0]-size[0]+1:
        while j<sizeb[1]-size[1]+1:
            x=A[i:i+size[0],j:j+size[1]]
            a[i,j]=(x*B).sum()/(np.size(x))*2
            j=j+1
        i=i+1
        j=0

    return a

def Sample(size,A,N):
    y=[]
    sizeb=np.shape(A)
    k=0
    while k<N:
        i=random.randint(0,sizeb[0]-size[0]-1)
        j=random.randint(0,sizeb[1]-size[1]-1)
        y.append(A[i:i+size[0],j:j+size[1]])
        k=k+1
    return y

def SampleVer2(size, A, N, l):
    images = []
    sizeb = np.shape(A)
    k = 0
    while k<N:
        #imageRandom = []
        i=random.randint(0,sizeb[0]-size[0]-1)
        j=random.randint(0,sizeb[1]-size[1]-1)
        #imageRandom.append(A[i:i+size[0],j:j+size[1]])
        #imageRandom.append(l)
        images.append(A[i:i+size[0],j:j+size[1]])
        k=k+1

    return images
"""size=np.shape(data)
print(size[0])
i=0
data1=[]
while i<size[0]:
    data1.append(Pool(data[i]))
    print(i)
    i=i+1
np.save('data4.npy', data1)"""

#print(F.MaxPool2d(A))
"""a=np.load('w.npy')
Inter.Array2image(a[0],'w0')
size=np.shape(a)
Net=Net0.Network((size[0],size[1]))
Net.w_node.Value=a
data=np.load('data.npy')
print(Net.pre(data[0]))
print(Net.pre(data[1000]))
Inter.traking(Net,'screen','map')"""
