import sqlite3

class DBHandler():

    def __init__(self):
        self.__connection = None
        self.__connected = False
        self.__cursor = None

    def __connect(self):

        try:

            self.__connection = sqlite3.connect('database.db')
            self.__connected = True
            self.__cursor = self.__connection.cursor()

        except Exception as ex:
            self.__connected = False
            print(ex)

            raise

    
    def __disconnect(self):

        try:

            self.__cursor.close()
            self.__connection.close()

        except Exception:

            raise

        finally:
            self.__connected = False

    
    def execute(self, query, data):

        row_id = 0

        try:

            self.__connect()

            self.__cursor.execute(query, data)

            row_id = self.__cursor.lastrowid

            self.__connection.commit()

        except Exception as ex:

            print("query=", query)
            print("data=", data)

            raise
        
        finally:

            self.__disconnect()
        
        return row_id
    
    def executeQuery(self, query, data=None):
        
        rows = None
        
        try:

            self.__connect()

            if data is not None:
                self.__cursor.execute(query, data)
            else:
                self.__cursor.execute(query)

            rows = self.__cursor.fetchall()



        except Exception as ex:

            print("query=", query)
            print("data=", data)

            raise
        
        finally:

            self.__disconnect()
        
        return rows

    
    def insertRows(self, query, rows):

        try:

            self.__connect()

            self.__cursor.executemany(query, rows)

            self.__connection.commit()

        except Exception as ex:

            print("query=", query)

            raise
        
        finally:

            self.__disconnect()
