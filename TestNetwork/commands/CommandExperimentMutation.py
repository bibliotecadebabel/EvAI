import children.pytorch.MutateNetwork as MutateNetwork
import children.pytorch.Network as nw
from DAO.database.dao import TestDAO, TestResultDAO
from DNA_Graph import DNA_Graph

class CommandExperimentMutation():

    def __init__(self, space, dataGen, testName, cuda=False):
        self.__space = space
        self.__cuda = cuda
        self.__dataGen = dataGen
        self.__bestNetwork = None
        self.__generateNetworks()
        self.__testName = testName
        self.__testDao = TestDAO.TestDAO()
        self.__testResultDao = TestResultDAO.TestResultDAO()
        
    def __generateNetworks(self):
        
        self.__networks = []
        self.__nodes = []

        space = self.__space
        CUDA = self.__cuda

        nodeCenter = self.__getNodeCenter()

        centerAdn = space.node2key(nodeCenter)

        centerNetwork = None
        if self.__bestNetwork is None:
            #print("new network")
            centerNetwork = nw.Network(centerAdn, cudaFlag=CUDA)
        else:
            print("new center (cloning network)=", self.__bestNetwork.adn)
            centerNetwork = self.__bestNetwork.clone()
        
        self.__nodes.append(nodeCenter)
        self.__networks.append(centerNetwork)

        for nodeKid in nodeCenter.kids:
            kidADN = space.node2key(nodeKid)
            kidNetwork = MutateNetwork.executeMutation(centerNetwork, kidADN)
            self.__nodes.append(nodeKid)
            self.__networks.append(kidNetwork)

    def __saveEnergy(self):
        
        for network in self.__networks:

            for node in self.__nodes:
                
                nodeAdn = self.__space.node2key(node)

                if str(nodeAdn) == str(network.adn):
                    node.objects[0].objects[0].energy = network.total_value
    
    def __getEnergyNode(self, node):

        return node.objects[0].objects[0].energy
    
    def __getNodeCenter(self):
        space = self.__space

        nodeCenter = None
        for node in space.objects:

            nodeAdn = space.node2key(node)

            if str(nodeAdn) == str(space.center):
                nodeCenter = node
        
        return nodeCenter


    def execute(self, periodSave, periodNewSpace, totalIterations, dt):
        
        dataGen = self.__dataGen

        test_id = self.__testDao.insert(testName=self.__testName, periodSave=periodSave, dt=dt, total=totalIterations)


        #print("center ADN= ", self.__space.node2key(self.__getNodeCenter()))

        for j in range(1, totalIterations+1):

            for network in self.__networks:
                network.Training(data=dataGen.data[0], labels=dataGen.data[1], dt=dt, p=1)
            
            if j % periodSave == 0:
                print("saving Energy, j=", j)
                self.__saveEnergy()
                self.__testResultDao.insert(idTest=test_id, iteration=j, dna_graph=self.__space)
        
            if j % periodNewSpace == 0:

                if self.__defineNewCenter() == True:
                    self.__generateNewSpace()
                    self.__generateNetworks()

    
    def __getBestNetwork(self):
        lowestEnergy = 10000
        bestNetwork = None
        for network in self.__networks:

            if network.total_value <= lowestEnergy:
                bestNetwork = network
                lowestEnergy = network.total_value

        return bestNetwork
    
    def __defineNewCenter(self):

        value = False
        newBestNetwork = self.__getBestNetwork()

        if self.__bestNetwork is None:
            
            value = True
            self.__bestNetwork = newBestNetwork
        
        else:

            if str(newBestNetwork.adn) != str(self.__bestNetwork.adn):
                value = True
                self.__bestNetwork = newBestNetwork
        
        return value

    def __generateNewSpace(self):
        oldSpace = self.__space

        newSpace = DNA_Graph(center=self.__bestNetwork.adn, size=oldSpace.size, dim=(oldSpace.x_dim, oldSpace.y_dim), 
                                condition=oldSpace.condition, typos=oldSpace.typos, type_add_layer=oldSpace.version)
        
        self.__space = None
        self.__space = newSpace




            
    
