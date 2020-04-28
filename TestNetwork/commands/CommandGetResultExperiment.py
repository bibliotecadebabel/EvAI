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
        
    # totalIteration = 0 => Get all results
    def execute(self, periodIteration, totalIteration=0):
        results = []
        self.__value = []
        
        if totalIteration == 0:
            results = self.__testResultDao.find(idTest=self.__testId)
        else:
            results = self.__testResultDao.findByLimitIteration(idTest=self.__testId, limitIteration=totalIteration)
        
        for result in results:

            if result.iteration % periodIteration == 0:
                self.__value.append(result)


    def getReturnParam(self):

        return self.__value


            
    
