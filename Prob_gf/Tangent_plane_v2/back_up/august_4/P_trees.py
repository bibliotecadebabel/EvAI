import Node as nd

def P_tree(Node,period,layers):
    if layers==0:
        pass
    else:
        i=0
        while i<period:
            Node.kids.append(P_tree(nd.Node(),period,layers-1))
            i=i+1
    return Node

def Op(Ope,tree):
    Ope(tree)
    if not(tree.kids==[]):
        for kid in tree.kids:
            Op(Ope,kid)

def Leaves(Node,y=None):
    if y==None:
        y=[]
    if Node.kids==[]:
        y.append(Node)
    else:
        for kid in Node.kids:
            Leaves(kid,y)
    return y


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



#positions = [g.recenter([-300,0],center),
#g.recenter([-200,100],center),
#g.recenter([-200,-100],center),
#g.recenter([-100,0],center),
#g.recenter([100,0],center),
#g.recenter([200,-100],center),
#g.recenter([200,100],center),
#g.recenter([300,0],center)]
