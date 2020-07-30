import Experiments.product_experiments  as experiments
import const.versions as directions_version

if __name__== '__main__':
    
    actions = ((0,1,0,0),(0,-1,0,0),(0,0,1,1),(0,0,-1,-1))
    test_name = "test_product_1"
    version = directions_version.POOL_VERSION
    experiments.run_cifar_user_input_bidi(test_name=test_name, mutations_actions=actions, const_direction_version=version)

