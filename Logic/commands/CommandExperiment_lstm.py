import children.pytorch.MutationManager as mutation_manager
import children.pytorch.NetworkDendrites as nw
from DAO.database.dao import TestDAO, TestResultDAO, TestModelDAO
from Geometric.Graphs.DNA_Graph import DNA_Graph as DNA_Graph
#from utilities.Abstract_classes.classes.uniform_random_selector_2 import centered_random_selector as random_selector
from Geometric.Creators.DNA_creators import Creator_from_selection as Creator_s
import const.path_models as const_path
import utilities.ExperimentSettings as ExperimentSettings
import os
import time
import utilities.FileManager as FileManager
import torch
import const.training_type as TrainingType
import utilities

class CommandExperimentCifar_Restarts():

    def __init__(self, settings : utilities.ExperimentSettings.ExperimentSettings):
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

        for network in self.__networks:

            for node in self.__nodes:

                nodeAdn = self.__space.node2key(node)

                if str(nodeAdn) == str(network.adn):
                    #network.generateEnergy(self.__settings.dataGen)
                    node.objects[0].objects[0].energy = network.getAcurracy()
                    print("saving energy network #", i, " - L=", network.getAcurracy())
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

    def __trainNetwork(self, network : nw.Network, dt_array, max_iter, keep_clone=False, allow_save_txt=False):

        dt_array_len = len(dt_array)
        avg_factor = dt_array_len // 4
        
        #network.generateEnergy(self.__settings.dataGen)
        #best_accuracy = network.getAcurracy()
        #print("initial accuracy= ", best_accuracy)

        for i in range(max_iter):
            print("iteration: ", i+1)
            network.iterTraining(self.__settings.dataGen, dt_array, self.__settings.ricap)
            #network.Training(data=self.__settings.dataGen, dt=dt_array, p=len(dt_array), full_database=True)
            network.generateEnergy(self.__settings.dataGen)
            current_accuracy = network.getAcurracy()
            print("current accuracy=", current_accuracy)

            if allow_save_txt == True and self.__settings.save_txt == True:
                loss = network.getAverageLoss(avg_factor)
                self.__fileManager.appendFile("iter: "+str(i+1)+" - Acc: "+str(current_accuracy)+" - Loss: "+str(loss))

        return network

    def execute(self):
        
        test_id = self.__testDao.insert(testName=self.__settings.test_name, dt=self.__settings.init_dt_array[0], 
                dt_min=self.__settings.init_dt_array[-1], batch_size=self.__settings.batch_size)

        if self.__settings.disable_mutation == False:

            self.__generateNewSpace(firstTime=True)
            self.__generateNetworks()

            for j in range(1, self.__settings.epochs+1):

                print("---- EPOCH #", j)

                for i  in range(1, len(self.__networks)):
                    print("Training net #", i, " - direction: ", self.__actions[i-1])
                    self.__networks[i] = self.__trainNetwork(network=self.__networks[i], dt_array=self.__settings.joined_dt_array, max_iter=self.__settings.max_joined_iter)

                self.__saveEnergy()
                self.__testResultDao.insert(idTest=test_id, iteration=j, dna_graph=self.__space)

                self.__bestNetwork = self.__getBestNetwork()

                self.__saveModel(network=self.__bestNetwork, test_id=test_id, iteration=j)

                self.__generateNewSpace()
                self.__generateNetworks()

                torch.cuda.empty_cache()


    def __getBestNetwork(self):

        highest_accuracy = -1
        bestNetwork = None
        for network in self.__networks[1:]:

            #network.generateEnergy(self.__settings.dataGen)
            #print("network accuracy=", network.getAcurracy())
            if network.getAcurracy() >= highest_accuracy:
                bestNetwork = network
                highest_accuracy = network.getAcurracy()

        print("bestnetwork= ", bestNetwork.getAcurracy())
        return bestNetwork

    def __generateNewSpace(self, firstTime=False):
        oldSpace = self.__space
        newCenter = self.__bestNetwork.adn

       
            
        #self.__selector = random_selector(num_actions=self.__selector.num_actions, directions=self.__selector.version,
        #                                    condition=self.__selector.condition, mutations=self.__selector.mutations)
            
        if firstTime == True:
            self.__selector.update(newCenter)
        else:
            self.__selector.update(oldSpace, newCenter)

        predicted_actions = self.__selector.get_predicted_actions()
       
        self.__actions = []
        self.__actions = predicted_actions
        stop = False
        
        while stop == False:

            newSpace = DNA_Graph(center=newCenter, size=oldSpace.size, dim=(oldSpace.x_dim, oldSpace.y_dim),
                                    condition=oldSpace.condition, typos=predicted_actions,
                                    type_add_layer=oldSpace.version, creator=Creator_s)

            nodeCenter = self.__getNodeCenter(newSpace)

            if len(nodeCenter.kids) >= 0:
                print("kids: ", len(nodeCenter.kids))
                stop = True

            '''
            if len(nodeCenter.kids) >= self.__selector.num_actions:
                print("num actions: ", self.__selector.num_actions)
                print("kids: ", len(nodeCenter.kids))
                time.sleep(2)
                stop = True
            else:
                print("num actions: ", self.__selector.num_actions)
                print("kids: ", len(nodeCenter.kids))
                
                for node in nodeCenter.kids:
                    direction = node.objects[0].objects[0].direction
                    print("direction accepted: ", direction)

                print("CURRENT CENTER: ", newCenter)
                raise Exception
            '''


        self.__space = None
        self.__space = newSpace

    def __saveModel(self, network, test_id, iteration):

        #network.generateEnergy(self.__settings.dataGen)
        fileName = str(test_id)+"_"+self.__settings.test_name+"_model_"+str(iteration)
        final_path = os.path.join("saved_models","cifar", fileName)

        dna = str(network.adn)
        accuracy = network.getAcurracy()

        network.saveModel(final_path)

        self.__testModelDao.insert(idTest=test_id,dna=dna,iteration=iteration,fileName=fileName, model_weight=accuracy, 
                                training_type=TrainingType.POST_TRAINING)

        print("model saved with accuarcy= ", accuracy)
