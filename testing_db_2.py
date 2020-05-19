from DAO.database.dao import TestDAO, TestModelDAO, TestResultDAO

test = TestDAO.TestDAO(db='database.db')
test_model = TestModelDAO.TestModelDAO(db='database.db')
test_result = TestResultDAO.TestResultDAO(db='database.db')

test.insert(dt=1, periodCenter=1, periodSave=1, total=20, testName="testing-db")
test_model.insert(idTest=1, dna="testing", iteration=1, fileName="Testing.file", model_weight=1)