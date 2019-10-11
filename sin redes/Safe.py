def safe_append(element,list):
    if element in list:
        pass
    else:
        list.append(element)

def safe_remove(element,list):
    if not(element in list):
        pass
    else:
        list.remove(element)

def safe_ope(key,dict,ope,x=None):
    if key in dict.keys():
        if x==None:
            ope(key,dict)
        else:
            ope(key,dict,x)






#positions = [g.recenter([-300,0],center),
#g.recenter([-200,100],center),
#g.recenter([-200,-100],center),
#g.recenter([-100,0],center),
#g.recenter([100,0],center),
#g.recenter([200,-100],center),
#g.recenter([200,100],center),
#g.recenter([300,0],center)]
