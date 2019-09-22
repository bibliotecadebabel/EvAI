import Node as nd
import P_trees as tr


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
