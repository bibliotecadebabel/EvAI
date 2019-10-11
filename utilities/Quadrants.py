import utilities.Node as nd
import utilities.P_trees as tr

class Quadrant():
    def __init__(self,shape):
        self.shape=shape
        self.objects=[]

def Divide_node(Node):
        n=len(Node.objects[0].shape)
        In=Indexes(n)
#        print(In)
        shape=Node.objects[0].shape
        if Node.kids==[]:
            pass
        else:
#            print('The number of kids is')
#            print(len(Node.kids))
            for  i in range(len(Node.kids)):
#                print('The value of i is')
#                print(i)
                kids=Node.kids
                kids[i].objects.append(Quadrant([]))
                Indi=In[i]
                j=0
                while j in range(n):
#                    print('The value of j is')
#                    print(j)
                    delta=(shape[j][1]-shape[j][0])/2
#                    print(delta)
                    kids[i].objects[0].shape.append(
                        [shape[j][0]+delta*Indi[j],
                            shape[j][0]+delta*(1+Indi[j])])
#                    print(kids[i].objects[0].shape)
                    j=j+1

def Divide(Node,layers):
    n=2**len(Node.objects[0].shape)
    tr.P_tree(Node,n,layers)
    tr.Op(Divide_node,Node)


def Index_node(Node):
    if Node.kids==[]:
        pass
    else:
        Node.kids[0].objects=Node.objects.copy()
        Node.kids[0].objects.append(0)
        Node.kids[1].objects=Node.objects.copy()
        Node.kids[1].objects.append(1)


def Indexes(n):
    a=nd.Node()
    tr.P_tree(a,2,n)
    tr.Op(Index_node,a)
    c=tr.Leaves(a)
    d=[]
    for Node in c:
        d.append(Node.objects)
    return d

def Print(Node):
    #print(Node.objects[0].shape)
    print(Node.objects[0].shape)


def Pnt(tree):
    tr.Op(Print,tree)

def In(p,Node):
    u=True
    for i in  range(len(Node.objects[0].shape)):
        side=Node.objects[0].shape[i]
        u=u and ((p[i]>=side[0])
            and (p[i]<side[1]))
    return u

def Find(p,tree):
    y=tr.Find_node(p,tree,In)
    return y


"""a=nd.Node()
b=Quadrant([[0,1],[0,1],[0,1]])
a.objects.append(b)
Divide(a,5)
#tr.Op(Print,a)
print('Testing in')
b=tr.Find_node([.9,.9,0.1],a,In)
print(b.objects[0].shape)"""
#print('done')
#a=nd.Node()
#a.objects.append(Quadrant([[0,1],[0,1]]))
#Divide(a,1)
#tr.Op(Range,a)
#tr.Op(Print,a)

#a.objects.append([0,1])
#a.kids[0].objects.append([0,0.5])
#a.kids[1].objects.append([0.5,1])



#positions = [g.recenter([-300,0],center),
#g.recenter([-200,100],center),
#g.recenter([-200,-100],center),
#g.recenter([-100,0],center),
#g.recenter([100,0],center),
#g.recenter([200,-100],center),
#g.recenter([200,100],center),
#g.recenter([300,0],center)]
