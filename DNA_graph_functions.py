import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane

def add_node(g,DNA):
    node=nd.Node()
    q=qu.Quadrant(DNA)
    p=tplane.tangent_plane()
    node.objects.append(q)
    q.objects.append(p)
    g.add_node(DNA,node)

def node2plane(node):
    q=node.objects[0]
    return q.objects[0]

def node2num_particles(node):
    p=node2plane(node)
    return p.num_particles

def node2direction(node):
    p= node2plane(node)
    return p.direction

def set_num_particles(node,particles):
    p=node2plane(node)
    p.num_particles=particles
