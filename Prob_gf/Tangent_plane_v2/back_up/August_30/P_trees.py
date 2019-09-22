import Node as nd

#Given a node Node
#Creates a periodic tree  with a number of layers determined
#by layers and period given by period
#The given tree is attached to Node

def P_tree(Node,period,layers):
    if layers==0:
        pass
    else:
        i=0
        while i<period:
            Node.kids.append(P_tree(nd.Node(),period,layers-1))
            i=i+1
    return Node

#Foward propagates operation OP on the tree

def Op(Ope,tree):
    Ope(tree)
    if not(tree.kids==[]):
        for kid in tree.kids:
            Op(Ope,kid)

# Returns the leaves on the tree whose father is node

def Leaves(Node,y=None):
    if y==None:
        y=[]
    if Node.kids==[]:
        y.append(Node)
    else:
        for kid in Node.kids:
            Leaves(kid,y)
    return y

#According to the bollean function Is_in_node,
#returns the node that contains p

def Find_node(p,tree,Is_in_node):
    for kid in tree.kids:
        if not(Is_in_node(p,kid)):
            pass
        else:
            if kid.kids==[]:
                y=kid
            else:
                y=Find_node(p,kid,Is_in_node)
    return y

#Determines if p is in tree according to definition Is_in_node.

def In_tree(p,tree,Is_in_node):
    u=False
    if tree==[]:
        pass
    else:
        u=u or  Is_in_node(p,tree)
        if tree.kids==[] or u:
            pass
        else:
            for kid in tree.kids:
                u=u or In_tree(p,kid,Is_in_node)
    return u





#Determines if node is a Node in tree
# Returns true or false accordingly.

def Node_In_tree(node,tree):
    u=False
    if tree==[]:
        pass
    else:
        u=u or  (tree==node)
        if tree.kids==[] or u:
            pass
        else:
            for kid in tree.kids:
                u=u or  Node_In_tree(node,kid)
    return u







#positions = [g.recenter([-300,0],center),
#g.recenter([-200,100],center),
#g.recenter([-200,-100],center),
#g.recenter([-100,0],center),
#g.recenter([100,0],center),
#g.recenter([200,-100],center),
#g.recenter([200,100],center),
#g.recenter([300,0],center)]
