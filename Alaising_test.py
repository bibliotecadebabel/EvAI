from utilities.Abstract_classes.classes.Alaising_cosine import (
    Alaising, Damped_Alaising)

def get_Damped_test_ep():
    Alai=Damped_Alaising(Max_iter=200000)
    print(Alai.initial_period)
    for k in range(200):
        increments=Alai.get_increments(2000)
        print(f'epoc {k}: dt_i= {increments[0]},f dt_f={increments[1]}')
    #print(Alai.get_increments(-n))
    return


def get_Damped_test():
    Alai=Damped_Alaising(initial_max=0.1,final_max=0.05,
        initial_min=10**(-99),final_min=10**(-99),Max_iter=100)
    print(Alai.initial_period)
    increments=Alai.get_increments(225)
    for k in range(224):
        print(f'epoc {k}: dt_i= {increments[k]},f dt_f={increments[k+1]}')
    #print(Alai.get_increments(-n))


def Damped_test():
    Alai=Damped_Alaising()
    print(Alai.initial_period)

def get_increment_log(n):
    Alai=Alaising()
    for k in range(5*n):
        print(Alai.get_increments(0))
        Alai.update()
    return

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
#get_increment_log(5)
#update_test()
#compilation_test()
