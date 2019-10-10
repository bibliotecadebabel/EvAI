import numpy as np



#Given coordinates and a frame, indicating the origin and the
#base it outputs standard coordinates of the vector if fram2=None.
#otherwise it outputs frame2coordinates of the vector

def coord2vector(cor,frame,frame2=None):
    if frame2==None:
        v_frame=np.asarray(cor,dtype=np.float64)
        v=np.zeros(len(v_frame),dtype=np.float64)
        for i in range(len(frame)):
            if i==0:
                v=v+np.asarray(frame[0],dtype=np.float64)

            else:
                v=v+v_frame[i-1]*np.asarray(frame[i]
                    ,dtype=np.float64)
        return v.tolist()
    else:
        v=vector2Ocoord(coord2vector(cor,frame),frame2)
        return v



#Given a vector and an orthonormal frame,
#it returns the coordinates of the vector
#in the given frame

def vector2Ocoord(v,framex):
    frame=framex.copy()
    x_0=np.asarray(frame[0],dtype=np.float64)
    v=np.asarray(v,dtype=np.float64)
    frame.pop(0)
    P=np.asarray(frame,dtype=np.float64)
    for i in range(len(P)):
        P[i]=P[i]/(P[i]*P[i]).sum()
    return (np.matmul(P,v-x_0)).tolist()

#remark np.asarray takes one row at the time
#as opposed to one colum at the time
