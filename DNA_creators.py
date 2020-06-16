import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane


class Creator():

    def __init__(self,typos,condition,type_add_layer=None):
        if type_add_layer==None:
            from DNA_directions import directions
        elif type_add_layer=='inclusion':
            from DNA_directions_i import directions,directions_labels
            self.directions_labels=directions_labels
        else:
            from DNA_directions_f import directions,directions_labels
            self.directions_labels=directions_labels
        self.typos=typos
        self.condition=condition
        self.directions=[]
        self.type_add_layer=type_add_layer
        for typo in typos:
            direction=directions.get(typo)
            print('the initial value of typo is')
            print(typo)
            if direction:
                self.directions.append(direction)
                typo=tuple(i * (-1) for i in typo)
                print('the final value of typo is')
                print(typo)
                direction=directions.get(typo)
                print('the final value of direction is')
                print(direction)
                if direction:
                    self.directions.append(direction)
        print('The number of directions is')
        print(len(self.directions))
        print('The value of typos is:')
        print(self.typos)

    def add_node(self,g,DNA):
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)

    def node2plane(self,node):
        q=node.objects[0]
        p=q.objects[0]
        return p

    def create(self,center,size,g=None):
        condition=self.condition
        if size>0:
            if isinstance(center,tuple):
                g=gr.Graph()
                self.add_node(g,center)
                center=g.key2node.get(center)
                self.create(center,size,g)
            else:
                q=center.objects[0]
                DNA_o=q.shape
                node_o=center
                for direction in self.directions:
                    if DNA_o:
                        for k in range (len(DNA_o)-2):
    #                        print('The value of k is')
    #                        print(k)
    #                        print('The value of DNA_o is:')
    #                        print(DNA_o)
                            DNA_f=condition(direction(k,DNA_o))
                            if DNA_f:
                                self.add_node(g,DNA_f)
                                node_f=g.key2node.get(DNA_f)
                                g.add_edges(DNA_o,[DNA_f])
                                if self.type_add_layer:
                                    label=self.directions_labels.get(direction)
                                    node=g.key2node.get(DNA_f)
                                    p=self.node2plane(node)
                                    if not (p.direction):
                                        p.direction=(k,label)
                if center.kids:
                    for kid in center.kids:
                        self.create(kid,size-1,g)
        return g

class Creator_from_selection_nm():

    def __init__(self,typos,condition,type_add_layer=None,num_morphisms=1,
        Selector=None):
        if type_add_layer==None:
            from DNA_directions import directions
        elif type_add_layer=='inclusion':
            from DNA_directions_i import directions,directions_labels
            self.directions_labels=directions_labels
        elif type_add_layer=='duplicate':
            from DNA_directions_duplicate import directions,directions_labels
            self.directions_labels=directions_labels
        elif type_add_layer=='clone':
            from DNA_directions_clone import directions,directions_labels
            self.directions_labels=directions_labels
        elif type_add_layer=='pool':
            from DNA_directions_pool import directions,directions_labels
            self.directions_labels=directions_labels
        elif type_add_layer=='h':
            from DNA_directions_h import directions,directions_labels
            self.directions_labels=directions_labels
        elif type_add_layer=='convex':
            from DNA_directions_convex import directions,directions_labels
            self.directions_labels=directions_labels
        else:
            from DNA_directions_f import directions,directions_labels
            self.directions_labels=directions_labels
        self.typos=typos
        self.condition=condition
        self.directions=directions
        self.type_add_layer=type_add_layer
        self.num_morphisms=num_morphisms
        self.Selector=Selector


    def add_node(self,g,DNA):
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)

    def node2plane(self,node):
        q=node.objects[0]
        p=q.objects[0]
        return p

    def create(self,center,size,g=None):
        selector=self.Selector
        condition=self.condition
        num_actions=selector.num_actions
        if size>0:
            if isinstance(center,tuple):
                g=gr.Graph()
                self.add_node(g,center)
                center=g.key2node.get(center)
                self.create(center,size,g)
            else:
                q=center.objects[0]
                DNA_o=q.shape
                node_o=center
                for l in range(num_actions):
                    if DNA_o:
                        DNA_f=DNA_o
                        label=[]
                        path=[]
                        for m in range (self.num_morphisms):
                            selector.update(DNA_f)
                            stop = False
                            while stop == False:
                                selector.update_predicted_actions()
                                actions=selector.get_predicted_actions()
                                typo=actions[0]
                                direction=self.directions.get(typo[1])
                                num_layer=typo[0]
                                temp=condition(direction(num_layer,DNA_f))
                                if isinstance(temp, tuple) == True:
                                    DNA_f = temp
                                    stop = True

                            label.append(typo)
                            path.append(DNA_f)
                            #print(f'The label is {label}')
                        if DNA_f:
                            self.add_node(g,DNA_f)
                            node_f=g.key2node.get(DNA_f)
                            g.add_edges(DNA_o,[DNA_f])
                            if self.type_add_layer:
                                node=g.key2node.get(DNA_f)
                                p=self.node2plane(node)
                                p.direction=label
                                p.path=path
                            #print(f'The label is{label}')
                            #print(f'The label is{len(p.path)}')
                if center.kids:
                    for kid in center.kids:
                        self.create(kid,size-1,g)

        return g

class Creator_from_selection():

    def __init__(self,typos,condition,type_add_layer=None):
        if type_add_layer==None:
            from DNA_directions import directions
        elif type_add_layer=='inclusion':
            from DNA_directions_i import directions,directions_labels
            self.directions_labels=directions_labels
        elif type_add_layer=='duplicate':
            from DNA_directions_duplicate import directions,directions_labels
            self.directions_labels=directions_labels
        elif type_add_layer=='clone':
            from DNA_directions_clone import directions,directions_labels
            self.directions_labels=directions_labels
        elif type_add_layer=='pool':
            from DNA_directions_pool import directions,directions_labels
            self.directions_labels=directions_labels
        elif type_add_layer=='h':
            from DNA_directions_h import directions,directions_labels
            self.directions_labels=directions_labels
        elif type_add_layer=='convex':
            from DNA_directions_convex import directions,directions_labels
            self.directions_labels=directions_labels
        else:
            from DNA_directions_f import directions,directions_labels
            self.directions_labels=directions_labels
        self.typos=typos
        self.condition=condition
        self.directions=directions
        self.type_add_layer=type_add_layer


    def add_node(self,g,DNA):
        node=nd.Node()
        q=qu.Quadrant(DNA)
        p=tplane.tangent_plane()
        node.objects.append(q)
        q.objects.append(p)
        g.add_node(DNA,node)

    def node2plane(self,node):
        q=node.objects[0]
        p=q.objects[0]
        return p

    def create(self,center,size,g=None):
        condition=self.condition
        if size>0:
            if isinstance(center,tuple):
                g=gr.Graph()
                self.add_node(g,center)
                center=g.key2node.get(center)
                self.create(center,size,g)
            else:
                q=center.objects[0]
                DNA_o=q.shape
                node_o=center
                for typo in self.typos:
                    if DNA_o:
                        for k in range (len(DNA_o)-2):
                            if k == typo[0]:
        #                       print('The value of k is')
        #                       print(k)
        #                       print('The value of DNA_o is:')
        #                       print(DNA_o)
                                direction=self.directions.get(typo[1])
                                if direction:
                                    DNA_f=condition(direction(k,DNA_o))
                                    if DNA_f:
                                        self.add_node(g,DNA_f)
                                        node_f=g.key2node.get(DNA_f)
                                        g.add_edges(DNA_o,[DNA_f])
                                        if self.type_add_layer:
                                            label=typo
                                            node=g.key2node.get(DNA_f)
                                            p=self.node2plane(node)
                                            if not (p.direction):
                                                p.direction=label
                if center.kids:
                    for kid in center.kids:
                        self.create(kid,size-1,g)
        return g
