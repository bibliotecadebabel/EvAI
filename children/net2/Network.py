import children.net2.Node as nd
import children.net2.Layer as ly
import children.net2.Functions as Functions

class Network:

    def __init__(self, objects):
        # objects [x, y, k]
        # x -> dimension x
        # y -> dimension y
        # k -> cantidad filtros
        self.objects = objects
        self.nodes = []
        self.__createStructure()
        self.total_value = 0

    def __createStructure(self):
        nodes = []

        nodes.append(nd.Node())
        nodes.append(nd.Node())
        nodes.append(nd.Node())
        nodes.append(nd.Node())
        nodes.append(nd.Node())

        nodes[0].kids.append(nodes[1])
        nodes[1].kids.append(nodes[2])
        nodes[2].kids.append(nodes[3])
        nodes[3].kids.append(nodes[4])

        nodes[1].parents.append(nodes[0])
        nodes[2].parents.append(nodes[1])
        nodes[3].parents.append(nodes[2])
        nodes[4].parents.append(nodes[3])


        self.nodes = nodes
        self.__assignLayers()

    def __assignLayers(self):

        self.nodes[0].objects.append(ly.createLayerA(self.nodes[0], self.objects))
        self.nodes[1].objects.append(ly.createLayerB(self.nodes[1], self.objects))
        self.nodes[2].objects.append(ly.createLayerC(self.nodes[2]))
        self.nodes[3].objects.append(ly.createLayerD(self.nodes[3]))
        self.nodes[4].objects.append(ly.createLayerE(self.nodes[4]))

    def assign(self, x, label=None):
        self.nodes[0].objects[0].value = x
        self.nodes[3].objects[0].label = label

    def Acumulate_der(self, n, peso=1):

        for i in range(len(self.nodes)):
            layer = self.nodes[i].objects[0]

            if layer.value_der is not None and layer.value_der_total is not None:
                layer.value_der_total += ((layer.value_der) / n) * peso

            if layer.bias_der is not None and layer.bias_der_total is not None:
                layer.bias_der_total += ((layer.bias_der) / n) * peso

            if layer.filter_der is not None and layer.filter_der_total is not None:
                layer.filter_der_total += ((layer.filter_der) / n) * peso

        self.total_value += (self.nodes[4].objects[0].value)/n

    def Regularize_der(self):

        for node in self.nodes:
            layer = node.objects[0]

            if layer.bias is not None and layer.bias_der_total is not None:
                layer.bias_der_total = layer.bias_der_total + layer.bias

            if layer.filters is not None and layer.filter_der_total is not None:
                layer.filter_der_total = layer.filter_der_total + layer.filters


        #self.total_value += Functions.Dot(self.nodes[0].objects[0].filters, self.nodes[0].objects[0].filters) + Functions.Dot(self.nodes[0].objects[0].bias, self.nodes[0].objects[0].bias) + Functions.Dot(self.nodes[1].objects[0].filters, self.nodes[1].objects[0].filters) + Functions.Dot(self.nodes[1].objects[0].bias, self.nodes[1].objects[0].bias)

        #print("regularize value total ", self.total_value)

    def Reset_der(self):

        for node in self.nodes:
            layer = node.objects[0]

            if layer.value_der is not None:
                layer.value_der = layer.value_der * 0

            if layer.bias_der is not None:
                layer.bias_der = layer.bias_der * 0

            if layer.filter_der is not None:
                layer.filter_der = layer.filter_der * 0

    def Reset_der_total(self):

        for node in self.nodes:
            layer = node.objects[0]

            if layer.value_der_total is not None:
                layer.value_der_total = layer.value_der_total * 0

            if layer.bias_der_total is not None:
                layer.bias_der_total = layer.bias_der_total * 0

            if layer.filter_der_total is not None:
                layer.filter_der_total = layer.filter_der_total * 0

        self.total_value = 0

    def Predict(self, image):
        self.assignLabels("c")
        self.nodes[0].objects[0].value = image[0]

        Functions.Propagation(self.nodes[4].objects[0])

        return self.nodes[3].objects[0].value

    def Training(self, data, dt=0.1, p=0.99):
        n = len(data) * 5/4
        peso = len(data) / 4

        #self.Train(data[0], peso, n)

        i=0
        while i < p:
            if i % 10==0:
                pass
                print(i)
                print(self.nodes[3].objects[0].value)
            self.Reset_der_total()
            self.Train(data[0], peso, n)

            for image in data[1:]:
                self.Train(image, 1, n)

            self.Regularize_der()
            self.Update(dt)
            i=i+1


    def Train(self, dataElement, peso, n):
        self.nodes[0].objects[0].value = dataElement[0]
        #self.nodes[3].objects[0].label = dataElement[1]

        self.assignLabels(dataElement[1])

        Functions.Propagation(self.nodes[4].objects[0])
        Functions.BackPropagation(self.nodes[0].objects[0])

        self.Acumulate_der(n, peso)

    def assignLabels(self, label):

        for node in self.nodes:
            node.objects[0].label = label

    def Update(self, dt):

        for node in self.nodes:
            layer = node.objects[0]

            #if layer.filter_der_total is not None and layer.bias_der_total is not None:

            if layer.filters is not None and layer.filter_der_total is not None:
                layer.filters = layer.filters - (layer.filter_der_total * dt)

            if layer.bias is not None and layer.bias_der_total is not None:
                layer.bias = layer.bias - (layer.bias_der_total * dt)

    def addFilters(self):

        ly.addFilterNodeA(self.nodes[0].objects[0])
        ly.addFilterNodeB(self.nodes[1].objects[0])
        self.objects[2]+=1

    def deleteFilters(self):

        ly.deleteFilterNodeA(self.nodes[0].objects[0])
        ly.deleteFilterNodeB(self.nodes[1].objects[0])
        self.objects[2]-=1

    def clone(self):

        network = Network(self.objects)


        for i in range(len(self.nodes)):

            layer = network.nodes[i].objects[0]

            layer.bias = Functions.np.copy(self.nodes[i].objects[0].bias)
            layer.filters = Functions.np.copy(self.nodes[i].objects[0].filters)

        return network
