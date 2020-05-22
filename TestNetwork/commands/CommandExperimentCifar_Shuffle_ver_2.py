import children.pytorch.MutateNetwork_Dendrites_clone as MutateNetwork
import children.pytorch.NetworkDendrites as nw
from DAO.database.dao import TestDAO, TestResultDAO, TestModelDAO
from DNA_Graph import DNA_Graph
from utilities.Abstract_classes.classes.random_selector import random_selector
from DNA_creator_duplicate_clone import Creator_from_selection_clone as Creator_s
import const.path_models as const_path
import TestNetwork.ExperimentSettings
import os

class CommandExperimentCifar_Restarts():

    def __init__(self, settings : TestNetwork.ExperimentSettings.ExperimentSettings):
        self.__space = settings.initial_space
        self.__selector = settings.selector
        
        self.__settings = settings
        
        self.__testDao = TestDAO.TestDAO(db='database.db')
        self.__testResultDao = TestResultDAO.TestResultDAO(db='database.db')
        self.__testModelDao = TestModelDAO.TestModelDAO(db='database.db')

        self.__bestNetwork = None

        self.__iterations_per_epoch = 0
        
        self.__bestNetwork = nw.Network(adn=settings.initial_dna, cudaFlag=settings.cuda,
                                momentum=settings.momentum, weight_decay=settings.weight_decay, 
                                enable_activation=settings.enable_activation)

        self.__bestNetwork_temp = None


    def __generateNetworks(self):

        self.__networks = []
        self.__nodes = []

        space = self.__space
        nodeCenter = self.__getNodeCenter(self.__space)

        centerNetwork = None

        print("new space's center: =", self.__bestNetwork.adn)
        centerNetwork = self.__bestNetwork

        self.__nodes.append(nodeCenter)
        self.__networks.append(centerNetwork)

        for nodeKid in nodeCenter.kids:
            kidADN = space.node2key(nodeKid)
            kidNetwork = MutateNetwork.executeMutation(centerNetwork, kidADN)
            self.__nodes.append(nodeKid)
            self.__networks.append(kidNetwork)

    def __saveEnergy(self):

        i = 0

        average_loss = 200

        for network in self.__networks:

            for node in self.__nodes:

                nodeAdn = self.__space.node2key(node)

                if str(nodeAdn) == str(network.adn):
                    node.objects[0].objects[0].energy = network.getAverageLoss(average_loss)
                    print("saving energy network #", i, " - L=", network.getAverageLoss(average_loss))
                    i +=1

    def __getEnergyNode(self, node):

        return node.objects[0].objects[0].energy

    def __getNodeCenter(self, space):

        nodeCenter = None
        for node in space.objects:

            nodeAdn = space.node2key(node)

            if str(nodeAdn) == str(space.center):
                nodeCenter = node

        return nodeCenter

    def __trainNetwork(self, network : nw.Network, dt_array, max_iter):

        best_network = network.clone()
        
        best_network.generateEnergy(self.__settings.dataGen)
        best_accuracy = best_network.getAcurracy()

        print("best current accuracy= ", best_network.getAcurracy())

        for i in range(max_iter):
            print("iteration: ", i+1)
            network.Training(data=self.__settings.dataGen, dt=dt_array, p=len(dt_array), full_database=True)
            network.generateEnergy(self.__settings.dataGen)
            current_accuracy = network.getAcurracy()

            print("current accuracy=", current_accuracy)
            if current_accuracy >= best_accuracy:
                del best_network
                best_accuracy = current_accuracy
                best_network = network.clone()

            else:
                print("interrupted, lower accuracy.")
                break
        
        best_network.generateEnergy(self.__settings.dataGen)
        print("final best accuarcy=", best_network.getAcurracy())
        return best_network


    def execute(self):

        dataGen = self.__settings.dataGen
        self.__iterations_per_epoch = len(dataGen._trainoader)

        test_id = self.__testDao.insert(testName=self.__settings.test_name, periodSave=self.__settings.period_save_space, 
                                    dt=self.__settings.init_dt_array[0], total=self.__settings.epochs, 
                                    periodCenter=self.__settings.period_new_space)


        print("TRAINING INITIAL NETWORK")
        self.__bestNetwork = self.__trainNetwork(network=self.__bestNetwork, 
                    dt_array=self.__settings.init_dt_array, max_iter=self.__settings.max_init_iter)

        self.__bestNetwork = self.__trainNetwork(network=self.__bestNetwork, 
                    dt_array=self.__settings.best_dt_array, max_iter=self.__settings.max_best_iter)

        self.__bestNetwork_temp=self.__bestNetwork.clone()
        self.accuracy_temp=self.__bestNetwork_temp.getAcurracy()

        self.__saveModel(self.__bestNetwork, test_id=test_id, iteration=0)

        self.__generateNewSpace()
        self.__generateNetworks()

        for j in range(1, self.__settings.epochs+1):

            print("---- EPOCH #", j)

            for i  in range(len(self.__networks)):
                print("Training net #", i)
                
                self.__networks[i] = self.__trainNetwork(network=self.__networks[i], dt_array=self.__settings.joined_dt_array,
                                        max_iter=self.__settings.max_joined_iter)


            self.__saveEnergy()
            self.__testResultDao.insert(idTest=test_id, iteration=j, dna_graph=self.__space)

            value = self.__getBestNetwork()
            self.__bestNetwork = value[0]
            newCenter = value[1]


            if newCenter == True:
                print("TRAINING BEST NETWORK")

                self.__bestNetwork = self.__trainNetwork(network=self.__bestNetwork, dt_array=self.__settings.best_dt_array,
                                            max_iter=self.__settings.max_best_iter)

                self.__saveModel(network=self.__bestNetwork, test_id=test_id, iteration=j)
            
            self.__generateNewSpace()
            self.__generateNetworks()
            


    def __getBestNetwork(self):
        highest_accuracy = -1
        bestNetwork = None
        print('NEW VERSION')
        newCenter = True
        for network in self.__networks:

            network.generateEnergy(self.__settings.dataGen)
            print("network accuracy=", network.getAcurracy())
            if network.getAcurracy() >= highest_accuracy:
                bestNetwork = network
                highest_accuracy = network.getAcurracy()

        if highest_accuracy>self.accuracy_temp:
            
            newCenter = True
            print("bestNetwork accuracy=", highest_accuracy)
            
            self.__bestNetwork_temp = bestNetwork.clone()

            self.accuracy_temp = highest_accuracy

            return [bestNetwork, newCenter]

        else:
            
            newCenter = False
            print("bestNetwork accuracy=", self.accuracy_temp)

            print('Best network did not change')

            return  [self.__bestNetwork_temp.clone(), newCenter]

    def __generateNewSpace(self):
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

    def __saveModel(self, network, test_id, iteration):

        network.generateEnergy(self.__settings.dataGen)
        fileName = str(test_id)+"_"+self.__settings.test_name+"_model_"+str(iteration)
        final_path = os.path.join("saved_models","cifar", fileName)

        dna = str(network.adn)
        accuracy = network.getAcurracy()

        network.saveModel(final_path)

        self.__testModelDao.insert(idTest=test_id,dna=dna,iteration=iteration,fileName=fileName, model_weight=accuracy)
        print("model saved with accuarcy= ", accuracy)
