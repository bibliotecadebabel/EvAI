import children.pytorch.MutateNetwork_Dendrites as MutateNetwork
import children.pytorch.NetworkDendrites as nw
from DAO.database.dao import TestDAO, TestResultDAO, TestModelDAO
from DNA_Graph import DNA_Graph
from utilities.Abstract_classes.classes.random_selector import random_selector
from DNA_creators import Creator_from_selection as Creator_s
import const.path_models as const_path
import os

class CommandExperimentCifar():

    def __init__(self, space, dataGen, testName, selector, cuda=False):
        self.__space = space
        self.__cuda = cuda
        self.__dataGen = dataGen
        self.__bestNetwork = None
        self.__generateNetworks()
        self.__testName = testName
        self.__testDao = TestDAO.TestDAO()
        self.__selector = selector
        self.__testResultDao = TestResultDAO.TestResultDAO()
        self.__testModelDao = TestModelDAO.TestModelDAO()

    def __generateNetworks(self):

        self.__networks = []
        self.__nodes = []

        space = self.__space
        CUDA = self.__cuda

        nodeCenter = self.__getNodeCenter(self.__space)

        centerAdn = space.node2key(nodeCenter)

        centerNetwork = None
        if self.__bestNetwork is None:
            print("new network")
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

    def __getNodeCenter(self, space):

        nodeCenter = None
        for node in space.objects:

            nodeAdn = space.node2key(node)

            if str(nodeAdn) == str(space.center):
                nodeCenter = node

        return nodeCenter


    def execute(self, periodSave, periodNewSpace, periodSaveModel, totalIterations, dt):

        dataGen = self.__dataGen

        print("inserting 1")
        test_id = self.__testDao.insert(testName=self.__testName, periodSave=periodSave, dt=dt, 
                                           total=totalIterations, periodCenter=periodNewSpace)


        #print("center ADN= ", self.__space.node2key(self.__getNodeCenter()))

        for j in range(1, totalIterations+1):
                
            print("epoch #", j)
            i = 0
            for network in self.__networks:
                print("training net #", i)
                network.TrainingCosineLR(dataGenerator=dataGen, dt=dt, epochs=1)
                i += 1
                
            self.__saveEnergy()
            self.__testResultDao.insert(idTest=test_id, iteration=j, dna_graph=self.__space)

            if j % periodNewSpace == 0:

                if self.__defineNewCenter() == True:
                    self.__generateNewSpace()
                    self.__generateNetworks()
            
            if j % periodSaveModel == 0:

                print("saving model")
                self.__saveModel(test_id=test_id, iteration=j)



    def __getBestNetwork(self):
        highest_accuracy = -1
        bestNetwork = None
        for network in self.__networks:
            
            network.generateEnergy(self.__dataGen)
            print("network accuracy=", network.getAcurracy())
            if network.getAcurracy() >= highest_accuracy:
                bestNetwork = network
                highest_accuracy = network.getAcurracy() 

        print("bestNetwork accuracy=", bestNetwork.getAcurracy())

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
        newCenter = self.__bestNetwork.adn

        newSpace = DNA_Graph(center=newCenter, size=oldSpace.size, dim=(oldSpace.x_dim, oldSpace.y_dim),
                                condition=oldSpace.condition, typos=oldSpace.typos, 
                                type_add_layer=oldSpace.version, creator=Creator_s)

        self.__space = None
        self.__space = newSpace
        
    def __generateNewSpace_backup(self):
        oldSpace = self.__space
        newCenter = self.__bestNetwork.adn

        stop = False
        while stop == False:
            
            self.__selector = random_selector()
            self.__selector.update(newCenter)
            predicted_actions = self.__selector.get_predicted_actions()

            newSpace = DNA_Graph(center=newCenter, size=oldSpace.size, dim=(oldSpace.x_dim, oldSpace.y_dim),
                                    condition=oldSpace.condition, typos=predicted_actions, 
                                    type_add_layer=oldSpace.version, creator=Creator_s)

            nodeCenter = self.__getNodeCenter(newSpace)

            if len(nodeCenter.kids) > 0:
                stop = True

        
        self.__space = None
        self.__space = newSpace

    def __saveModel(self, test_id, iteration):

        fileName = str(test_id)+"_"+self.__testName+"_model_"+str(iteration)
        final_path = os.path.join("saved_models","cifar",fileName)
        self.__bestNetwork.generateEnergy(self.__dataGen)
        dna = str(self.__bestNetwork.adn)
        
        self.__bestNetwork.saveModel(final_path)
        
        self.__testModelDao.insert(idTest=test_id,dna=dna,iteration=iteration,fileName=fileName, model_weight=self.__bestNetwork.getAcurracy())