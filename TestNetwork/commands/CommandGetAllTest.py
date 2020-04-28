
from DAO.database.dao import TestDAO

class CommandGetAllTest():

    def __init__(self):

        self.__testDao = TestDAO.TestDAO()
        self.__value = None
        
    def execute(self):

        self.__value = []

        self.__value = self.__testDao.findAll()

    def getReturnParam(self):

        return self.__value
        
        




            
    
