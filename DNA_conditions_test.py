from DNA_conditions import max_layer, max_filter


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



#DNA_Creator_s(11,11)
DNA_max_filter(11,11)
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
