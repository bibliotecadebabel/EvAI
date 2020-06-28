import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.P_trees as tr
import utilities.Graphs as gr

def add_edgest_test():
    g=gr.Graph()
    g.add_node(0,nd.Node())
    g.add_node(1,nd.Node())
    g.add_node(2,nd.Node())
    g.add_edges(1,[0])
    g.add_edges(2,[1])
    print('The kids are')
    print(len(g.key2node[0].kids))
    print(len(g.key2node[1].kids))
    print(len(g.key2node[2].kids))
