
from DAO.database.dao import test_dao

class CommandGetAllTest():

    def __init__(self, db="database_product.db"):

        self.__testDao = test_dao.TestDAO(db)
        self.__value = None
        
    def execute(self):

        self.__value = []

        self.__value = self.__testDao.findAll()

    def getReturnParam(self):

        return self.__value
        
        




            
    
