from DAO.database.dao import TestDAO, TestResultDAO


testDao = TestDAO.TestDAO()
testResult = TestResultDAO.TestResultDAO()

test_name = "test-1"
dt = 0.01
iterations = 5000
period_save = 5
period_newspace  = 1

test_id = testDao.insert(testName=test_name, periodSave=5, dt=dt, total=5000, periodCenter=1)

for i in range(1000):
    
    if i % 5 == 0:

        testResult.insert(idTest=test_id, iteration=i, dna_graph=newSpace)
