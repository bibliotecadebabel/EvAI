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

def Test_node_3(network,n=5,dt=0.01):
    k=0
    layer_f=network.nodes[4].objects[0]
    layer_i=network.nodes[3].objects[0]
    while k<5:
        print(layer_f.value)
        network.Reset_der_total()
        print('The value of the parent is')
        print(layer_f.node.parents[0].objects[0].value)
        layer_f.propagate(layer_f)
        layer_i.backpropagate(layer_i)
        network.Acumulate_der()
        network.Update(dt)
        k=k+1



x = 2
y = 2
k = 2



objects = Functions.np.full((3), (x, y, k))

network = nw.Network([x,y,k])
print('testing node 3')
Test_node_3(network)


#data = []

#generateData(data, objects, 100)

#network.Training(data=data, dt=0.05, p=0.9)
