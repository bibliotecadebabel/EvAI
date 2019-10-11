import time
import random
import Status as St
import json


def readFile():
    status = St.Status()
    try:
        file = open("body.txt",'r')
        value = file.read()
        status.jsonToObject(json.loads(value))
        file.close()
    except Exception:
        status = None
        #print(Exception)
    
    return status

while True:

    status = readFile()

    if status is None:
        print("HI")