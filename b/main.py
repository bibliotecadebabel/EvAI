import pygame, sys
import Quadrants as qu
import Node as nd
import P_trees as tr
n=4
sectors=nd.Node()


pygame.init()


display_width = 800
display_height = 600

sectors.objects.append(qu.Quadrant([[0,display_width],
    [0,display_height]]))
qu.Divide(sectors,n)
#qu.Pnt(sectors)

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('A bit Racey')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()
crashed = False
carImg = pygame.image.load('drone.png')
carImg=pygame.transform.scale(carImg, (int(display_width/(2**n)),
        int(display_height/(2**n))))

def car(x,y):
    gameDisplay.blit(carImg, (x,y))

x =  (display_width * 0.45)
y = (display_height * 0)

def Set_size(Image,x,y):
    y=pygame.transform.scale(Image, (x,y))
    return y


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
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    key = pygame.key.get_pressed()
    if key[pygame.K_UP]:
        y=y+1


    pygame.display.flip()
    #frame=frame+1
