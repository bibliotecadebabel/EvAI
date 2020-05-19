from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising, Damped_Alaising)

def get_Damped_test():
    Alai=Damped_Alaising(Max_iter=10)
    print(Alai.initial_period)
    print(Alai.get_increments(100,update=False))
    #print(Alai.get_increments(-n))
    return

def Damped_test():
    Alai=Damped_Alaising()
    print(Alai.initial_period)


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

#Damped_initialiciation_test()
get_Damped_test()

#update_test()
#compilation_test()
