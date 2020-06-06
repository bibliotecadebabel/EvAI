from DNA_conditions import max_layer, max_filter
import test_DNAs as DNAs
import DNA_conditions


def DNA_conv_last():
    DNA=DNAs.DNA_ep_85_5_pool
    print(DNA_conditions.max_filter_dense(DNA,400))

def DNA_max_pool_layer_test():
    DNA=DNAs.DNA_calibration_2
    print(DNA_conditions.max_pool_layer(DNA,10))

def DNA_min_filter(x,y):
    DNA=((-1, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y), (1, 5, 2), (2,))
    print(DNA_conditions.min_filter(DNA,8))

def DNA_max_filter(x,y):
    center=((0, 3, 5, 3, 3),(0, 8, 8, 3,3),(0,11,5, x, y), (1, 5, 2), (2,))
    print('The DNA is')
    print(center)
    M_filter=30
    print('M_filter is')
    print(M_filter)
    print('The output of max filter is')
    print(max_filter(center,M_filter))
    M_filter=8
    print('but when M_filter is')
    print(M_filter)
    print('The output of max filter is')
    print(max_filter(center,M_filter))


def max_parents_test():
    DNA=DNAs.DNA_h
    print(DNA_conditions.max_parents(DNA,1))

def max_kernel_dense_test():
    DNA=DNAs.DNA_h
    print(DNA_conditions.max_kernel_dense(DNA,5))

def dict2condition_test():
    list_conditions={DNA_conditions.max_filter : 256,
            DNA_conditions.max_filter_dense : 256,
            DNA_conditions.max_kernel_dense : 9,
            DNA_conditions.max_layer : 30,
            DNA_conditions.min_filter : 3,
            DNA_conditions.max_pool_layer : 5,
            DNA_conditions.max_parents : 2}
    def condition(DNA):
        return DNA_conditions.dict2condition(DNA,dict)
    DNA=DNAs.DNA_calibration_2
    print(condition(DNA))

def max_filter_test():
    DNA=DNAs.DNA_calibration_2
    print(DNA_conditions.max_filter(DNA,530))

DNA_conv_last()
#max_filter_test()
#dict2condition_test()
#max_kernel_dense_test()
#max_parents_test()
#DNA_Creator_s(11,11)
#DNA_max_filter(11,11)
#layer_increase_i(11,11)
#kernel_increase_i(11,11)
#add_filter_i(11,11)
#linear_filter_creator(11,11)
#linear_filter(11,11)
#   linear_kernel_depth(11,11)
#kernel_height_creator(11,11)
#kernel_height_creator_i(11,11)
#linear_filter_new(11,11)
#linear_kernel_width_new(11,11)
#linear_filter(11,11)
#linear_kernel_width(11,11)

#We have dictionary of types of
#DNA that select the creator
#from the type in the initializations
#then expand from the center
#the types must contain, set of directions, which is is pair of cell centere
#and mutation tpe
