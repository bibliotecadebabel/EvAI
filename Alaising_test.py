from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising)

def get_increment(n):
    Alai=Alaising()
    print(Alai.get_increments(n))
    print(Alai.get_increments(-n))
    print(Alai.get_increments(0))
    return

def update_test():
    Alai=Alaising()
    Alai.update()
    print(Alai.time)

def compilation_test():
    Alai=Alaising()
    print('done')

get_increment(5)
#update_test()
#compilation_test()
