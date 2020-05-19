from DAO.database.DBHandler import DBHandler
from Entities.TestModelEntity import TestModelEntity
import json

class TestModelDAO():

    def __init__(self, db='database_product.db'):
        self.__handler = DBHandler(db)
    
    def insert(self, idTest, dna, iteration, fileName, model_weight):

        query = """INSERT INTO test_models(id_test, dna, iteration, model_name, model_weight) values(?,?,?,?,?);"""

        data = (idTest, dna, iteration, fileName, model_weight)

        self.__handler.execute(query, data)


    def find(self, idTest, iteration):
        
        query = """SELECT * FROM test_models WHERE id_test = ? AND iteration = ?"""
        data = (idTest,iteration)

        row = self.__handler.executeQuery(query, data)

        model_params = TestModelEntity()

        model_params.load(data=row)

        return model_params

    def delete(self, id):
        pass

    def findAll(self, idTest):
        pass

    def deleteAll(self):
        pass