from DAO.database.dao import TestDAO

test = TestDAO.TestDAO()
while True:
    test.insert(dt=1, periodCenter=1, periodSave=1, total=10, testName="testing-product-default")