import numpy as np
import copy

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


class Damped_Alaising():
    def __init__(self,min=0.0001,
        initial_max=0.2,
        final_max=0.001,
        initial_min=0.0001,
        final_min=0.00001,
        Max_iter=2000):
        self.initial_max=initial_max
        self.final_max=final_max
        self.current_max=0
        self.current_min=0
        self.initial_min=initial_min
        self.final_min=final_min
        self.Max_iter=Max_iter
        self.time=0
        self.initial_period=int(np.log(Max_iter)/np.log(2)-1)+1
        self.current_period=self.initial_period
        self.current_max=initial_max
        self.current_min=initial_min
        self.local_time=0

    def get_increments(self,size):
        m=self.current_min
        M=self.current_max
        t_M=self.current_period
        t_o=self.time
        if size>0:
            Alai=copy.deepcopy(self)
            output=[]
            for i in range(size):
                output.append(Alai.get_increments(0))
                Alai.update()
            return output
        elif size<0:
            Alai=copy.deepcopy(self)
            output=[]
            for i in range(-size):
                output.append(Alai.get_increments(0))
                Alai.rewind()
            ouput=output.reverse()
            return output
        return m+1/2*(M-m)*(1+np.cos(t_o/t_M*np.pi))

    def update_amplitude(self):
        n=self.Max_iter
        t=self.time
        m_o=np.log(self.initial_min)
        M_o=np.log(self.initial_max)
        m_f=np.log(self.final_min)
        M_f=np.log(self.final_max)
        self.current_min=10**(m_o+t*(m_f-m_o)/n)
        self.current_max=10**(M_o+t*(M_f-M_o)/n)
        self.time+=1

    def update(self):
        self.update_amplitude()
        if self.local_time>self.current_period:
           self.current_period=2*self.current_period
           self.local_time=0
        else:
           self.local_time+=1
        self.time+=1

    def rewind(self):
        self.update_amplitude()
        if self.local_time<0:
           self.current_period=1/2*self.current_period
           self.local_time-=self.current_period
        else:
           self.local_time-=1
           self.time-=1
