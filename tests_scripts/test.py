import children.net2.Network as nw
import children.net2.Functions as Functions
#from DAO import GeneratorFromImage
from utilities import Data_generator
import children.Interfaces as Inter
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

def Test_node_3(network,n=5,dt=0.001):
    k=0
    layer_f=network.nodes[4].objects[0]
    layer_i=network.nodes[3].objects[0]
    while k<5:
        #print("K= ",k)
        network.Reset_der_total()
        #print('The value of the parent is ', layer_f.node.parents[0].objects[0].value, "\n")
        layer_f.propagate(layer_f)
        layer_i.backPropagate(layer_i)
        network.Acumulate_der(n)
        layer_i.value+=-layer_i.value_der*dt
        #network.Update(dt)
        print("value of layer_f: ", layer_f.value)
        k=k+1

def Test_node_2(network,label="c",n=5,dt=0.001):
    k=0
    layer_f=network.nodes[4].objects[0]
    layer_i=network.nodes[2].objects[0]
    layer_i.label=label
    while k<5:
        network.Reset_der_total()
        j=3
        while j<5:
            layer=network.nodes[j].objects[0]
            layer.propagate(layer)
            j=j+1
        layer_i.backPropagate(layer_i)
        network.Acumulate_der(n)
        layer_i.value+=-layer_i.value_der*dt
        #network.Update(dt)
        print("value of layer_f: ", layer_f.value)
        k=k+1

def Test_node_1(network,n=5,dt=0.1):
    k=0
    layer_f=network.nodes[4].objects[0]
    layer_i=network.nodes[1].objects[0]
    while k<5:
        network.Reset_der_total()
        j=2
        while j<5:
            layer=network.nodes[j].objects[0]
            layer.propagate(layer)
            j=j+1
        layer_i.backPropagate(layer_i)
        network.Acumulate_der(1)
        network.Update(dt)
        print("value of layer_f: ", layer_f.value)
        k=k+1

def Test_node_0(network,n=1000,dt=0.1):
    k=0
    layer_f=network.nodes[4].objects[0]
    layer_i=network.nodes[0].objects[0]
    while k<n:
        #network.Predict(layer_i.value)
        network.assignLabels("n")
        network.Reset_der_total()
        #j=1
        #while j<5:
            #layer=network.nodes[j].objects[0]
            #layer.propagate(layer)
            #j=j+1

        #layer_i.backPropagate(layer_i)
        Functions.Propagation(layer_f)
        Functions.BackPropagation(layer_i)
        network.Acumulate_der(1)
        network.Update(dt)
        print(network.total_value)
        #print("value of layer_f: ", layer_f.value)
        #print("value of layer_f: ", layer_f.node.parents[0].objects[0].value)
        #print("value_der nodo 3: ", network.nodes[3].objects[0].value_der)
        k=k+1

def Test_training(network,data):
    print('The label is:')
    print(dataGen.Data[1][1])
    network.Training(data=dataGen.Data, dt=0.1, p=1)
    print(network.total_value)
    pass

def Test_multipleNetworks():
    networks = []

    dataGen = Data_generator.Data_gen()
    dataGen.S=200
    dataGen.gen_data()
    size = dataGen.size
    ks = [1, 3, 5, 8, 10]


    for i in range(5):
        networks.append(nw.Network([size[0], size[1], ks[i]]))

    for k in range(500):
        for i in range(len(networks)):
            networks[i].Training(data=dataGen.Data, dt=0.001, p=1)
            print("value network #", str(i),": ", networks[i].total_value)

    for j in range(len(networks)):
        networkName = "network-"+str(j)+"-map"
        Inter.trak(networks[i], dataGen.A, networkName)



def Test_modifyNetwork(network, data):

    print("Entrenando red \n")
    network.Training(data=data, dt=0.01, p=200)
    print("mutando la red: Agregando Filtro \n")
    network.addFilters()
    print("Entrenando la red mutada \n")
    network.Training(data=data, dt=0.001, p=100)
    print("mutando la red: Eliminando Filtro \n")
    network.deleteFilters()
    print("Entrenando la red mutada \n")
    network.Training(data=data, dt=0.001, p=100)

def Test_cloneNetwork(network, data):

    for i in range(100):
        network.Training(data=data, dt=0.01, p=10)
        clone = network.clone()
        print("entreno red clonada")
        clone.Training(data=data, dt=0.01, p=10)
        print("prob clone network: ", clone.nodes[3].objects[0].value)
        print("agrego filtro")
        clone.addFilters()
        clone.addFilters()
        clone.addFilters()
        print("entreno red clonada")
        clone.Training(data=data, dt=0.01, p=10)
        print("prob clone network: ", clone.nodes[3].objects[0].value)
        clone.deleteFilters()
        print("elimino filtro red clonada y entreno")
        clone.Training(data=data, dt=0.01, p=10)
        print("prob clone network: ", clone.nodes[3].objects[0].value)
z

dataGen = Data_generator.Data_gen()
dataGen.S=200

dataGen.gen_data()

print(dataGen.size)
x = dataGen.size[0]
y = dataGen.size[1]
k = 4



objects = Functions.np.full((3), (x, y, k))

network = nw.Network([x,y,k])

#Test_cloneNetwork(network, dataGen.Data)

#Test_training(network,dataGen.Data)

Test_multipleNetworks()

#Test_node_0(network)

#Test_modifyNetwork(network, dataGen.Data)
#data = []

#generateData(data, objects, 100)

#Test_multipleNetworks()

"""

print('testing node 3')
Test_node_3(network)
print('testing node 2 label=c')
Test_node_2(network)
print('testing node 2 label=n')
Test_node_2(network,"n")
print('testing node 1')
Test_node_1(network)
print('testing node 1')

"""
#Test_node_0(network)






#network.Training(data=data, dt=0.05, p=10)
