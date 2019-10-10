import Node as nd
import P_trees as tr
from Safe import safe_append, safe_remove, safe_ope

class Graph():
    def __init__(self):
        self.key2node={}
        self.node2key={}
        self.objects=[]
        self.directed=False
    def add_node(self,key,node):
        if not(key in self.key2node.keys()):
            self.key2node.update({key : node})
            self.node2key.update({node : key})
            self.objects.append(node)
    def add_edges(self,key,keys):
        if not(key in self.key2node.keys()):
            pass
        else:
            node=self.key2node[key]
            for key_k in keys:
                if not(key_k in self.key2node.keys()):
                    pass
                else:
                    node_k=self.key2node[key_k]
                    safe_append(node_k,node.kids)
                    if self.directed==False:
                        safe_append(node,node_k.kids)
    def remove_edge(self,key_a,key_b):
        def rm_edge(key_a,dict,key_b):
            node=dict[key_a]
            safe_remove(dict.get(key_b),node.kids)
        safe_ope(key_a,self.key2node,rm_edge,key_b)
        safe_ope(key_b,self.key2node,rm_edge,key_a)
        pass
    def remove_node(self,key):
        if key in self.key2node.keys():
            node=self.key2node[key]
            self.objects.remove(node)
            for kid in node.kids:
                self.remove_edge(key,self.node2key[kid])
            self.key2node.pop(key, None)
            self.node2key.pop(node, None)

#Given a graph, returns a tree, whose
#objects are the nodes in a spanning tree
#of the graph in the case where n=None. If n==None then it produces the spaning tree
# of the graphs at n conectivity gegrees from the gr


def spanning_tree(Graph, tree=None, leave=None, n=None):
    if n == None:
        tree = spanning_tree_total(Graph)
    else:
        tree = spanning_tree_n(Graph, n=n)
    return tree

#Given a graph, returns a tree, whose
#objects are the nodes in a spanning tree
#of the graph

def spanning_tree_total(Graph,tree=None,leave=None):
    if tree==None:
        tree=nd.Node()
        tree.objects.append(Graph)
        if Graph.kids==[]:
            pass
        else:
            for kid in Graph.kids:
                if tr.In_tree(kid,tree,it_points):
                    pass
                else:
                    a=nd.Node()
                    tree.kids.append(a)
                    a.objects.append(kid)
                    b=a
                    spanning_tree_total(kid,tree,b)
    else:
        if Graph.kids==[]:
            pass
        else:
            for kid in Graph.kids:
                if tr.In_tree(kid,tree,it_points):
                    pass
                else:
                    a=nd.Node()
                    leave.kids.append(a)
                    a.objects.append(kid)
                    if not(kid.kids==[]):
                        spanning_tree_total(kid,tree,a)
    return tree

#Given a graph node and an integer, returns a tree, whose
#objects are the nodes in a spanning tree
#of the graph cibtining the n degrees of conectivity from
#the original graph node

def spanning_tree_n(Graph,tree=None,leave=None,n=None):
    if n==0:
        pass
    else:
        n=n-1
        if tree==None:
            tree=nd.Node()
            tree.objects.append(Graph)
            if Graph.kids==[]:
                pass
            else:
                for kid in Graph.kids:
                    if tr.In_tree(kid,tree,it_points):
                        pass
                    else:
                        a=nd.Node()
                        tree.kids.append(a)
                        a.objects.append(kid)
                        b=a
                        spanning_tree_n(kid,tree,b,n)
        else:
            if Graph.kids==[]:
                pass
            else:
                for kid in Graph.kids:
                    if tr.In_tree(kid,tree,it_points):
                        pass
                    else:
                        a=nd.Node()
                        leave.kids.append(a)
                        a.objects.append(kid)
                        if not(kid.kids==[]):
                            spanning_tree_n(kid,tree,a,n)
    return tree

def it_points(node1,node2):
    return node2.objects[0]==node1

G=Graph()
for k in range(10):
    G.add_node(k,nd.Node())
G.add_edges(1,[2,3,3,7])
G.remove_node(3)
print(len(G.key2node[7].kids))
print('Hi')
a=[3]
safe_append(4,a)
safe_remove(5,a)
print(a)
