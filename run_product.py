import Experiments.product_experiments  as experiments
import const.versions as directions_version

if __name__== '__main__':
    
    actions = (
    (1,0,0,0),(1,0,0,0),
    (0,1,0,0),(0,1,0,0),
    (4,0,0,0),
    (0,0,1),(0,0,-1),
    (0,0,1,1),(0,0,-1,-1),
    (0,0,2)
    )
    version = directions_version.CONVEX_VERSION
    experiments.run_cifar_user_input_bidi(test_name=None, mutations_actions=actions, const_direction_version=version)