from DAO.database.DBHandler import DBHandler
import time
from Entities.TestEntity import TestEntity

class TestDAO():

    def __init__(self, db="database_product.db"):
        self.__handler = DBHandler(db)
    
    def insert(self, testName, dt, dt_min, batch_size, max_layers=0, max_filters=0, 
                max_filter_dense=0, max_kernel_dense=0, max_pool_layer=0, max_parents=0):
        
        query = """INSERT INTO test 
                    (name, dt, dt_min, batch_size, max_layers, max_filters, max_filter_dense,  
                        max_kernel_dense, max_pool_layer, max_parents, start_time) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""

        current_time = str(time.time())
        
        data = (testName, dt, dt_min, batch_size, max_layers, max_filters, max_filter_dense, max_kernel_dense,
                max_pool_layer, max_parents, current_time)

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
        
        query = """SELECT * FROM test order by id DESC"""

        rows = self.__handler.executeQuery(query)

        value = []
        
        for row in rows:
            
            testEntity = TestEntity()
            
            testEntity.load(row)

            value.append(testEntity)

        return value

    def deleteAll(self):
        pass