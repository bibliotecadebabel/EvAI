import pygame, sys
import Quadrants as qu
import Node as nd
import P_trees as tr
import Product as program
import tkinter as tk






class Button():
    def __init__(self,action=None,position=None,scale=None,image=None):
        self.action=action
        self.position=position
        self.objects=scale
        self.image=image



def dx(status):
    status.dx=int(user_input('Enter dyadic subdivision'))

def dt(status):
    status.dt=float(user_input('Enter value of dt'))

def num_particels(status):
    status.n=int(user_input('Enter number of particels'))

def beta(status):
    status.beta=float(user_input('Enter beta value'))

def initialize(status):
    program.initialize_parameters(status)

def run_stop(status):
    if status.active:
        status.active=False
    else:
        program.create_objects(status)
        status.active=True

actions =	{
  'run_stop' : run_stop,
  'initialize' : initialize,
  'beta' : beta,
  'num_particels' : num_particels,
  'dt' : dt,
  'dx' : dx,
}






# This process takes a list of the form
# [[1 2 3 'A'] [1 2 3 'B']  [1 2 3 'C']  ]
# where in teach list [1 2 3 'B'] the firs two number
# denote de position, third one scale and last None
# a string type denoting the keyword for the action of the Button
def load(tree,list):
    Nodes_button=[]
    display_width=tree.objects[0].shape[0][1]
    display_height=tree.objects[0].shape[1][1]
    for button in list:
        n=button[2]
        pos=[button[0],button[1]]
        scale=button[2]
        action=button[3]
        p=coordinate(pos,scale,tree)
        Lx=tree.objects[0].shape[0][1]
        Ly=tree.objects[0].shape[1][1]
        dx=Lx/(2**n)
        dy=Ly/(2**n)
        Node=qu.Find(p,tree)
        Node.objects[0].objects.append(Button(action=actions[action],image=
            pygame.image.load('drone.png')))
        button=Node.objects[0].objects[0]
        ima=button.image
        b=pygame.transform.scale(ima, (int(display_width/(2**n)),
            int(display_height/(2**n))))
        button.image=b
        Nodes_button.append(Node)
    return Nodes_button

def plot_buttons(Nodes_button,tree,Display):
    for Node in Nodes_button:
        image=Node.objects[0].objects[0].image
        pos=[]
        pos.append(Node.objects[0].shape[0][0])
        pos.append(Node.objects[0].shape[1][0])
        Display.blit(image, (pos[0],pos[1]))


def coordinate(pos,scale,tree):
    position=[0,0]
    lx=tree.objects[0].shape[0][1]
    ly=tree.objects[0].shape[1][1]
    dx=lx/(2**scale)
    dy=ly/(2**scale)
    position[0]=pos[0]*dx
    position[1]=ly-pos[1]*dy
    return position


class GetEntry():
    def __init__(self, master,message):
        self.master=master
        self.entry_contents=None
        self.e = tk.Entry(master)
        self.e.grid(row=0, column=0)
        self.e.focus_set()
#        master.bind('<Return>', self.callback)
        tk.Button(master, text=message, width=30,
               command=self.callback).grid(row=10, column=0)
    def callback(self):
        self.entry_contents=self.e.get()
        self.master.quit()


def user_input(message):
    master = tk.Tk()
    GE=GetEntry(master,message)
    master.mainloop()
    master.destroy()
    return GE.entry_contents





#program=[[3,3,4,'Show'],[5,3,4,'Remove']]
