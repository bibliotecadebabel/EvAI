import children.pytorch.MutateNetwork as MutateNetwork
import children.pytorch.Network as nw
from DAO.database.dao import TestDAO, TestResultDAO
from DNA_Graph import DNA_Graph

class CommandGetResultExperiment():

    def __init__(self, testId):
        self.__testId = testId
        self.__testDao = TestDAO.TestDAO()
        self.__testResultDao = TestResultDAO.TestResultDAO()
        self.__value = None
        
    def execute(self, periodIteration, minRange, maxRange):
        results = []
        self.__value = []

        results = self.__testResultDao.findByLimitIteration(idTest=self.__testId, minRange=minRange, maxRange=maxRange)
        
        self.__value = results


    def getReturnParam(self):

        return self.__value


            
    
