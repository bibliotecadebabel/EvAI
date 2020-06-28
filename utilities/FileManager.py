class FileManager:


    def __init__(self):
        self.__fileName = None
    
    def setFileName(self, name):
        self.__fileName = name

    def writeFile(self, value):

        try:
            file_target = open(self.__fileName, "w")
            file_target.write(value)
            file_target.close()
        except:
            print("Error writing file: ", self.__fileName)
    
    def appendFile(self, value):
       
        try:
            file_target = open(self.__fileName, "a")
            file_target.write(value+"\n")
            file_target.close()
        except:
            print("Error writing (append mode) file: ", self.__fileName)

    def readFile(self):
        
        try: 
            file_target = open(self.__fileName, "r")
            value = file_target.readlines()
        except:
            print("Error reading file: ", self.__fileName)
        
        return value