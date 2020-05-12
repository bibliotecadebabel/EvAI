from DAO import GeneratorFromImage
from DNA_Graph import DNA_Graph
from DNA_Phase_space import DNA_Phase_space
from timing import timing
import numpy as np

def update_from_select(self):
    phase_space=self.phase_space
    creator=self.Creator
    selector=self.Selector
    version=self.version
    if phase_space.max_changed:
        num_particles = phase_space.num_particles
        old_graph = phase_space.DNA_graph
        old_center= old_graph.center
        condition = old_graph.condition
        typos = old_graph.typos
        node_max = phase_space.node_max_particles
        center = phase_space.node2key(node_max)
        selector.update(old_graph,new_center=center)
        actions=selector.get_predicted_actions()
        x = old_graph.x_dim
        y = old_graph.y_dim
        space=DNA_Graph(center,1,(x,y),condition,actions,
            version,creator)
        phase_space.DNA_graph = space
        phase_space.objects = space.objects
        phase_space.support=[]
        phase_space.create_particles(num_particles+1)
        phase_space.attach_balls()
        phase_space.max_changed = False
        phase_space.node_max_particles = None
        self.space = space
        self.phase_space= phase_space
        self.objects=phase_space.objects
        self.support=phase_space.support
        self.Graph=phase_space.DNA_graph


def update_space_default(self):
    phase_space=self.phase_space
    if phase_space.max_changed:
        num_particles = phase_space.num_particles
        old_graph = phase_space.DNA_graph
        condition = old_graph.condition
        typos = old_graph.typos
        node_max = phase_space.node_max_particles
        center = phase_space.node2key(node_max)
        x = old_graph.x_dim
        y = old_graph.y_dim
        space=DNA_Graph(center,self.dx,(x,y),condition,typos,'inclusion')
        #Phase_space = DNA_Phase_space(space)
        phase_space.DNA_graph = space
        phase_space.objects = space.objects
        phase_space.support=[]
        phase_space.create_particles(num_particles+1)
        phase_space.attach_balls()
        phase_space.max_changed = False
        phase_space.node_max_particles = None
        self.space = space
        self.phase_space= phase_space
        self.objects=phase_space.objects
        self.support=phase_space.support
        self.Graph=phase_space.DNA_graph
