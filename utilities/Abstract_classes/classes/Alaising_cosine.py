import numpy as np

class Alaising():
    def __init__(self,min=0.0001,max=0.5,time=1,max_time=2000):
        self.max=max
        self.min=min
        self.max_time=max_time
        self.time=time
    def get_increments(self,size):
        m=self.min
        M=self.max
        t_M=self.max_time
        t_o=self.time
        if size>0:
            return [ m+1/2*(M-m)*(1+np.cos(t/t_M*np.pi))
                     for t in range(t_o,t_o+size)]
        elif size<0:
            return [ m+1/2*(M-m)*(1+np.cos(t/t_M*np.pi))
                     for t in range(t_o+size,t_o)]
        return m+1/2*(M-m)*(1+np.cos(t_o/t_M*np.pi))

    def update(self):
        self.time+=1
