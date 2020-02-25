import utilities.Quadrants as qu
import utilities.Node as nd
import utilities.Graphs as gr
import TangentPlane as tplane
import DNA_directions as dir

def linear_filter_add(x,y):
    center=((0, 3, 4, 2, 2),(0, 4,5, x-1, y-1), (1, 5, 2), (2,))
    print('linear_filter: done')
    print(center)
    print('has mutated to')
    print(dir.increase_filters(0,center))
    print('and')
    print(dir.increase_filters(1,center))
    return

def linear_filter_remove(x,y):
    center=((0, 3, 4, 2, 2),(0, 4,5, x-1, y-1), (1, 5, 2), (2,))
    print('linear_filter: done')
    print(center)
    print('has mutated to')
    print(dir.decrease_filters(0,center))
    print('and')
    print(dir.decrease_filters(1,center))
    return

def kernel_increase(x,y):
    center=((0, 3, 4, 2, 2),(0, 4,5, x-1, y-1), (1, 5, 2), (2,))
    print('linear_filter: done')
    print(center)
    print('has mutated to')
    print(dir.increase_kernel(0,center))
    print('and')
    print(dir.increase_kernel(1,center))
    return

def kernel_decrease(x,y):
    center=((0, 3, 4, 6,6),(0, 4,5, x-5, y-5), (1, 5, 2), (2,))
    print('linear_filter: done')
    print(center)
    print('has mutated to')
    print(dir.decrease_kernel(0,center))
    print('and')
    print(dir.decrease_kernel(1,center))
    return

#linear_filter_add(11,11)
#linear_filter_remove(11,11)
#kernel_increase(11,11)
kernel_decrease(11,11)
