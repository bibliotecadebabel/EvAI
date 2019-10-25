import numpy as np
from PIL import Image
import os
import random
import children.Net as Net0

def traking(Net,Name,Out_name):
    dir_path=os.path.dirname(os.path.realpath(__file__))
    im=Image.open(dir_path+'\\'+Name+'.png')
    size=np.shape(Net.w_node.Value[0])
    im0=im.convert('RGB')
    Ima=np.array(im0)
    sizeb=np.shape(Ima)
    i=0
    j=0
    a=np.zeros((sizeb[0], sizeb[1], 3), dtype=np.uint8)
    while i<sizeb[0]-size[0]:
        while j<sizeb[1]-size[1]:
            x=Ima[i:i+size[0],j:j+size[1]]
            p=Net.pre(x)
            a[i,j]=a[i,j]+p*255
            j=j+1
        i=i+1
        j=0
        if i % 10==0:
            print(i)
    Array2image(a,Out_name)

def trak(Net,Ima,Out_name):
    #size=np.shape(Net.w_node.Value[0])
    size= np.shape(Net.nodes[0].objects[0].value)
    sizeb=np.shape(Ima)
    print(sizeb)
    i=0
    j=0
    a=np.zeros((sizeb[0], sizeb[1], 3), dtype=np.uint8)
    while i<sizeb[0]-size[0]:
        while j<sizeb[1]-size[1]:
            x=Ima[i:i+size[0],j:j+size[1]]
            #p=Net.pre(x)
            image = []
            image.append(x)
            image.append("n")
            p = Net.Predict(image)
            a[i,j]=a[i,j]+p*255
            j=j+1
        i=i+1
        if i % 10==0:
            print(i/10)
        j=0
    Array2image(a,Out_name)

def trakPytorch(Net,Out_name, dataGen):
    #size=np.shape(Net.w_node.Value[0])
    Ima = dataGen.A
    labelCircle = dataGen.label[0]

    size= np.shape(Net.nodes[0].objects[0].value)
    sizeb=np.shape(Ima)

    i=0
    j=0
    #a=torch.ones([1,3, size[2], size[3]], dtype=torch.float32)
    a=np.zeros((sizeb[0], sizeb[1], 3), dtype=np.uint8)
    while i<sizeb[0]-size[2]:
        while j<sizeb[1]-size[3]:
            x=Ima[i:i+size[2],j:j+size[3]]
            x = dataGen.numpyToTorch(x)
            p = Net.Predict(x, labelCircle)
            #print("prob traking: ",p)
            a[i,j]=a[i,j]+p*255
            j=j+1
        i=i+1
        if i % 10==0:
            print(i/10)
        j=0
    print(a.shape)
    Array2image(a,Out_name)  
    

def Image2array(Name,type=None):
    dir_path=os.path.dirname(os.path.realpath(__file__))
    if type is None:
        im=Image.open(dir_path+'\\'+Name+'.png')
    else:
        im=Image.open(dir_path+'\\'+Name+'.'+type)
    im0= im.convert('RGB')
    im=im0
    pix_val = list(im.getdata())
    y=np.array(im)
    #y=np.reshape(np.array(pix_val),(im.size[0],im.size[1],3))
    #y=0
    return y
def Array2image(x,Name):
    Im=Image.fromarray(x,'RGB')
    Im.save(Name+'.png')

def Random_array(a,b):
    size=[a,b]
    c=np.zeros((size[0], size[1], 3), dtype=np.uint8)
    i=0
    j=0
    while i<size[0]:
        while j<size[1]:
            c[i,j]=[random.randint(1,256),random.randint(1,256),random.randint(1,256)]
            c[i,j]=c[i,j]
            j=j+1
        i=i+1
        j=0
    return c
