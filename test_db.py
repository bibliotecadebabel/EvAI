from DAO.database.dao import TestResultDAO, TestDAO
from DNA_Graph import DNA_Graph
from DNA_conditions import max_layer
from DNA_creators import Creator

testDao = TestDAO.TestDAO()
testResultDao = TestResultDAO.TestResultDAO()


def DNA_test_i(x,y):
    def condition(DNA):
        output=True
        if DNA:
            for num_layer in range(0,len(DNA)-3):
                layer=DNA[num_layer]
                x_l=layer[3]
                y_l=layer[4]
                output=output and (x_l<x) and (y_l<y)
        if output:
            return max_layer(DNA,3)
    center=((0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y), (1, 5, 2), (2,))
    version='inclusion'
    space=space=DNA_Graph(center,2,(x,y),condition,(0,(0,0,1,1),(0,1,0,0),(1,0,0,0)),version)

    return space


space = DNA_test_i(11,11)

idTest = 1

testDao.insert(idTest, "test-1")
testResultDao.insert(idTest=idTest, iteration=10, dna_graph=space)


test = testDao.find(idTest)
idTest = test[0][0]

results = testResultDao.find(idTest)

for result in results:
    print("dna=", result.dna)
    print("isCenter=", result.isCenter)
    print("direction=", result.tangentPlane.direction)
