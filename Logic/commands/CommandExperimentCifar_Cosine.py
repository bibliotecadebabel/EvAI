import children.pytorch.MutationManager as mutation_manager
import children.pytorch.NetworkDendrites as nw
from DAO.database.dao import TestDAO, TestResultDAO, TestModelDAO
from DNA_Graph import DNA_Graph
#from utilities.Abstract_classes.classes.uniform_random_selector_2 import centered_random_selector as random_selector
from DNA_creators import Creator_from_selection as Creator_s
import const.path_models as const_path
import TestNetwork.ExperimentSettings
import os
import time
import utilities.FileManager as FileManager
import torch

class CommandExperimentCifar_Restarts():

    def __init__(self, settings : TestNetwork.ExperimentSettings.ExperimentSettings):
        self.__space = settings.initial_space
        self.__selector = settings.selector
        self.__networks = []
        self.__nodes = []

        self.__settings = settings

        self.__testDao = TestDAO.TestDAO(db='database.db')
        self.__testResultDao = TestResultDAO.TestResultDAO(db='database.db')
        self.__testModelDao = TestModelDAO.TestModelDAO(db='database.db')

        self.__bestNetwork = None

        if self.__settings.loadedNetwork == None:
            self.__bestNetwork = nw.Network(adn=settings.initial_dna, cudaFlag=settings.cuda,
                                    momentum=settings.momentum, weight_decay=settings.weight_decay,
                                    enable_activation=settings.enable_activation,
                                    enable_track_stats=settings.enable_track_stats, dropout_value=settings.dropout_value,
                                    dropout_function=settings.dropout_function, enable_last_activation=settings.enable_last_activation,
                                    version=settings.version, eps_batchnorm=settings.eps_batchorm)
        else:
            self.__bestNetwork = self.__settings.loadedNetwork

        self.mutation_manager = mutation_manager.MutationManager(directions_version=settings.version)

        self.__actions = []

        self.__fileManager = FileManager.FileManager()

        if self.__settings.save_txt == True:
            self.__fileManager.setFileName(self.__settings.test_name)
            self.__fileManager.writeFile("")




    def __generateNetworks(self):

        networks = self.__networks
        del networks

        nodes = self.__nodes
        del nodes

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
            kidNetwork = self.mutation_manager.executeMutation(centerNetwork, kidADN)
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

    def __trainNetwork(self, network : nw.Network, dt_max, dt_min, epochs, restart_epochs, allow_save_txt=False):
        
        network.generateEnergy(self.__settings.dataGen)
        best_accuracy = network.getAcurracy()
        print("initial accuracy= ", best_accuracy)

        if allow_save_txt == True:
            network.trainingWarmRestarts(dataGenerator=self.__settings.dataGen, dt_max=dt_max, dt_min=dt_min, epochs=epochs, 
                                        restar_period=restart_epochs, ricap=self.__settings.ricap, evalLoss=False, 
                                        fileManager=self.__fileManager)
        else:
            network.trainingWarmRestarts(dataGenerator=self.__settings.dataGen, dt_max=dt_max, dt_min=dt_min, epochs=epochs, 
                                        restar_period=restart_epochs, ricap=self.__settings.ricap, evalLoss=False, 
                                        fileManager=None)

        network.generateEnergy(self.__settings.dataGen)
        current_accuracy = network.getAcurracy()
        print("final accuracy=", current_accuracy)

        return network

    def execute(self):

        test_id = self.__testDao.insert(testName=self.__settings.test_name, periodSave=self.__settings.period_save_space,
                                    dt=self.__settings.init_dt_max, total=self.__settings.epochs,
                                    periodCenter=self.__settings.period_new_space)


        print("TRAINING INITIAL NETWORK")
        print("Allow interrupts= ", self.__settings.allow_interupts)


        self.__bestNetwork = self.__trainNetwork(network=self.__bestNetwork, dt_max=self.__settings.init_dt_max, 
                            dt_min=self.__settings.init_dt_min, epochs=self.__settings.init_epochs, 
                            restart_epochs=self.__settings.init_restart_period, allow_save_txt=True)

        self.__bestNetwork = self.__trainNetwork(network=self.__bestNetwork, dt_max=self.__settings.best_dt_max, 
                            dt_min=self.__settings.best_dt_min, epochs=self.__settings.best_epochs, 
                            restart_epochs=self.__settings.best_restart_period, allow_save_txt=True)


        self.__saveModel(self.__bestNetwork, test_id=test_id, iteration=0)

        if self.__settings.disable_mutation == False:

            self.__generateNewSpace()
            self.__generateNetworks()

            for j in range(1, self.__settings.epochs+1):

                print("---- EPOCH #", j)

                for i  in range(1, len(self.__networks)):
                    print("Training net #", i, " - direction: ", self.__actions[i-1])
                    self.__networks[i] = self.__trainNetwork(network=self.__bestNetwork, dt_max=self.__settings.joined_dt_max, 
                            dt_min=self.__settings.joined_dt_min, epochs=self.__settings.joined_epochs, 
                            restart_epochs=self.__settings.joined_restart_period, allow_save_txt=False)

                self.__saveEnergy()
                self.__testResultDao.insert(idTest=test_id, iteration=j, dna_graph=self.__space)

                self.__bestNetwork = self.__getBestNetwork()

                print("TRAINING BEST NETWORK")

                self.__bestNetwork = self.__trainNetwork(network=self.__bestNetwork, dt_max=self.__settings.best_dt_max, 
                            dt_min=self.__settings.best_dt_min, epochs=self.__settings.best_epochs, 
                            restart_epochs=self.__settings.best_restart_period, allow_save_txt=True)

                self.__saveModel(network=self.__bestNetwork, test_id=test_id, iteration=j)

                self.__generateNewSpace()
                self.__generateNetworks()

                torch.cuda.empty_cache()


    def __getBestNetwork(self):

        highest_accuracy = -1
        bestNetwork = None
        for network in self.__networks[1:]:

            network.generateEnergy(self.__settings.dataGen)
            print("network accuracy=", network.getAcurracy())
            if network.getAcurracy() >= highest_accuracy:
                bestNetwork = network
                highest_accuracy = network.getAcurracy()

        print("bestnetwork= ", bestNetwork.getAcurracy())
        return bestNetwork

    def __generateNewSpace(self):
        oldSpace = self.__space
        newCenter = self.__bestNetwork.adn

        stop = False
        while stop == False:

            #self.__selector = random_selector(num_actions=self.__selector.num_actions, directions=self.__selector.version,
            #                                    condition=self.__selector.condition, mutations=self.__selector.mutations)
            self.__selector.update(newCenter)
            predicted_actions = self.__selector.get_predicted_actions()

            self.__actions = []
            self.__actions = predicted_actions

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
