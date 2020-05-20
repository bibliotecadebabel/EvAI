import DNA_conditions as program
import Bugs_direction_f as DNAs

def restrict_conections_test(DNA):
    print(program.restrict_conections(DNA))


restrict_conections_test(DNAs.DNA_init)
restrict_conections_test(DNAs.DNA_r)

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
