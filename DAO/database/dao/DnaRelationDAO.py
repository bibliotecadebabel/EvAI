from DAO.database.DBHandler import DBHandler

class DnaRelationDAO():

    def __init__(self):
        self.__handler = DBHandler()
    
    def insert(self, id, entity):
        self.__handler._connect()

        self.cursor = self.__handler.getCursor()

    def update(self, id, entity):
        pass

    def find(self, id):
        pass

    def delete(self, id):
        pass

    def findAll(self):
        pass

    def deleteAll(self):
        pass