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

#Foward propagates operation OP on the tree,accumulating intermididiates
#in cumulative if necessary

def Op(Ope,tree,cumulative=None):
    if cumulative==None:
        Ope(tree)
        if not(tree.kids==[]):
            for kid in tree.kids:
                Op(Ope,kid)
    else:
        Ope(tree,cumulative)
        if not(tree.kids==[]):
            for kid in tree.kids:
                Op(Ope,kid,cumulative)

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



#calculates the size of the given tree

def tree_size(tree):
    def add_one(node,cumulative):
        cumulative[0]=cumulative[0]+1
    cumulative=[0]
    Op(add_one,tree,cumulative)
    return cumulative[0]

#given a tree it produces a dictionary that to each node assings
# the a the shortest path to the node
def tree_paths(tree):
    paths={ tree : [tree]}
    def branch(node,paths):
        if node.kids==[]:
            pass
        else:
            path=paths[node]
            for kid in node.kids:
                new_path=path.copy()
                new_path.append(kid)
                paths.update({kid : new_path})
    Op(branch,tree,paths)
    return  paths
#given a tree it produces a dictionary that to each node assings
#the lenght of the minimal path towards the node

def tree_distances(tree):
    distances={}
    paths=tree_paths(tree)
    for node in paths:
     distances.update({node : len(paths[node])-1})
    return distances




#positions = [g.recenter([-300,0],center),
#g.recenter([-200,100],center),
#g.recenter([-200,-100],center),
#g.recenter([-100,0],center),
#g.recenter([100,0],center),
#g.recenter([200,-100],center),
#g.recenter([200,100],center),
#g.recenter([300,0],center)]
