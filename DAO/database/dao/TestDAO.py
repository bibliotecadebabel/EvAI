from DAO.database.DBHandler import DBHandler

class TestDAO():

    def __init__(self):
        self.__handler = DBHandler()
    
    def insert(self, idTest, testName):

        query = """INSERT INTO test (id, name) VALUES (?, ?);"""
        data = (idTest, testName)

        self.__handler.execute(query, data)

    def find(self, idTest):
        
        query = """SELECT * FROM test WHERE id = ?"""
        data = (idTest,)

        row = self.__handler.executeQuery(query, data)

        return row
    
    def findByName(self, name):
        
        query = """SELECT * FROM test WHERE name = ?"""
        data = (name,)

        row = self.__handler.executeQuery(query, data)

        return row

    def delete(self, id):
        pass

    def findAll(self):
        
        query = """SELECT * FROM test WHERE name = ?"""

        rows = self.__handler.executeQuery(query)

        return rows

    def deleteAll(self):
        pass