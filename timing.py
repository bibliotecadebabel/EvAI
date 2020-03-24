import time

def timing(program):
    start = time.time()
    program()
    print(time.time()- start)
