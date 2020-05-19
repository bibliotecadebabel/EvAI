from DAO.database.DBHandler import DBHandler

from Entities.TestEntity import TestEntity

class TestDAO():

    def __init__(self, db="database_product.db"):
        self.__handler = DBHandler(db)
    
    def insert(self, testName, periodSave, dt, total, periodCenter, enable_mutation=1):
        
        query = """INSERT INTO test (name, period, dt, total_iteration, period_center, enable_mutation) VALUES (?, ?, ?, ?, ?, ?);"""
        data = (testName, periodSave, dt, total, periodCenter, enable_mutation)

        row_id = self.__handler.execute(query, data)

        return row_id

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
        
        query = """SELECT * FROM test"""

        rows = self.__handler.executeQuery(query)

        value = []
        
        for row in rows:
            
            testEntity = TestEntity()
            
            testEntity.load(row)

            value.append(testEntity)

        return value

    def deleteAll(self):
        pass