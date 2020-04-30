import pygame, sys
import Quadrants as qu
import Node as nd
import P_trees as tr
import Buttons as bu
import Product as program



def car(x,y):
    gameDisplay.blit(carImg, (x,y))

#Creates status of the program object
Status=program.Status()
#Initializes the Status (later will be moved to buttons)
program.initialize_parameters(Status)
program.create_objects(Status)

pygame.init()

n=5
#display_width = 1500
#display_height = 700

display_width = 800
display_height = 800

#Initializes sectors and load buttons on them

sectors=nd.Node()
sectors.objects.append(qu.Quadrant([[0,display_width],
    [0,display_height]]))
qu.Divide(sectors,n)
Node_buttons=bu.load(sectors,[[3,3,n],[6,3,n],
    [9,3,n],[12,3,n]])



gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Gui')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()
crashed = False

#Cursor image (temporary)
carImg = pygame.image.load('drone.png')
carImg=pygame.transform.scale(carImg, (int(display_width/(2**n)),
        int(display_height/(2**n))))





while True:
    gameDisplay.fill(black)
    mouse_poss=pygame.mouse.get_pos()
    x=mouse_poss[0]
    y=mouse_poss[1]
    b=qu.Find([x,y],sectors)
    px=int(b.objects[0].shape[0][0])
    py=int(b.objects[0].shape[1][0])
    #print(b.objects[0].shape)
#    print(px)
#    print(x)
    car(px,py)
    bu.plot_buttons(Node_buttons,sectors,gameDisplay)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_poss = pygame.mouse.get_pos()
            x=mouse_poss[0]
            y=mouse_poss[1]
            b=qu.Find([x,y],sectors)
            if b.objects[0].objects==[]:
                pass
            else:
                button=b.objects[0].objects[0]
                button.action(Status)




    #key = pygame.key.get_pressed()
    #if key[pygame.K_UP]:
    #    y=y+1
    #ba.plot(pygame)

    #Update status
    if Status.active:
        program.update(Status)
        program.plot(Status,gameDisplay,[[6,28],[8,14],5],sectors)


    pygame.display.flip()
    #frame=frame+1
