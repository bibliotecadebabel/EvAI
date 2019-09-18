import Network as nw
import Functions
import decimal

decimal.getcontext().prec = 100

def generateData(data, objects, n):
    
        circulo = []
        circulo.append(Functions.np.zeros((objects[0], objects[1], 3), dtype=float))
        circulo.append("c")

        data.append(circulo)

        for i in range(objects[0]//2):
            for j in range(objects[1]):
                circulo[0][i][j] = [255, 255, 255]

        for i in range(objects[0]//2, objects[0]):
            for j in range(objects[1]):
                circulo[0][i][j] = [1,1,1]

        for i in range(n-1):
            imagenRandom = []
            imagenRandom.append(generateImageRandom(objects))
            imagenRandom.append("n")

            data.append(imagenRandom)


def generateImageRandom(objects):
    image = Functions.np.zeros((objects[0], objects[1], 3), dtype=float)

    
    for i in range(objects[0]):
        for j in range(objects[1]):
            image[i,j] = [Functions.random.randint(1, 255), 
                Functions.random.randint(1, 255), 
                Functions.random.randint(1, 255)]
    
    return image

x = 3
y = 2
k = 2


objects = Functions.np.full((3), (x, y, k))

network = nw.Network(objects)
data = []

generateData(data, objects, 100)

network.Training(data=data, dt=0.05, p=0.9)