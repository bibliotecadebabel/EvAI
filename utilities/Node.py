class Node():
    def __init__(self):
        self.kids=[]
        self.parents=[]
        self.objects=[]
    def attach(self,obj):
        self.objects.append(obj)
    def get_object(self):
        return self.objects[0]


#print('Hi')





#positions = [g.recenter([-300,0],center),
#g.recenter([-200,100],center),
#g.recenter([-200,-100],center),
#g.recenter([-100,0],center),
#g.recenter([100,0],center),
#g.recenter([200,-100],center),
#g.recenter([200,100],center),
#g.recenter([300,0],center)]
