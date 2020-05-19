from DAO.database.DBHandler import DBHandler
from Entities.TestResultEntity import TestResultEntity
import json

class TestResultDAO():

    def __init__(self, db='database_product.db'):
        self.__handler = DBHandler(db)
    
    def insert(self, idTest, iteration, dna_graph):

        query = """INSERT INTO test_result(id_test, iteration, dna, tangentPlane, center) values(?,?,?,?,?);"""

        rows = []

        for node in dna_graph.objects:

            dna =  DnaJson(node.objects[0].shape) # quadrant
            tangentPlane = node.objects[0].objects[0] # tangentPlane
            value = TangentPlaneJson(tangentPlane)
            json_tangent = str(json.dumps(value.__dict__))
            json_dna = str(json.dumps(dna.__dict__))

            isCenter = 0
            if str(node.objects[0].shape) == str(dna_graph.center): 
                isCenter = 1

            data = (idTest, iteration, json_dna, json_tangent, isCenter)

            rows.append(data)



        self.__handler.insertRows(query, rows)


    def find(self, idTest):
        
        query = """SELECT * FROM test_result WHERE center = 1 AND id_test = ?"""
        data = (idTest,)

        rows = self.__handler.executeQuery(query, data)

        value = []
        for row in rows:
            resultTest = TestResultEntity()

            resultTest.load(data=row)

            value.append(resultTest)
        
        return value

    def findByLimitIteration(self, idTest, minRange, maxRange):
        
        query = """SELECT * FROM test_result WHERE center = 1 AND id_test = ? AND iteration >= ? AND iteration <= ?"""
        data = (idTest, minRange, maxRange)

        rows = self.__handler.executeQuery(query, data)

        value = []
        for row in rows:
            resultTest = TestResultEntity()

            resultTest.load(data=row)

            value.append(resultTest)
        
        return value

    def delete(self, id):
        pass

    def findAll(self, idTest):
        pass

    def deleteAll(self):
        pass

class TangentPlaneJson():

    def __init__(self, tangentPlane):

        self.divergence=tangentPlane.divergence
        self.metric=tangentPlane.metric
        self.density=tangentPlane.density
        self.num_particles=tangentPlane.num_particles
        self.gradient=tangentPlane.gradient
        self.reg_density=tangentPlane.reg_density
        self.interaction_field=tangentPlane.interaction_field
        self.difussion_field=tangentPlane.difussion_field
        self.external_field=tangentPlane.external_field
        self.force_field=tangentPlane.force_field
        self.energy=tangentPlane.energy
        self.interaction_potential=tangentPlane.interaction_potential
        self.velocity_potential=tangentPlane.velocity_potential
        self.direction=tangentPlane.direction

class DnaJson():

    def __init__(self, dna):

        self.dna = dna