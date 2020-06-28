import Experiments.product_experiments  as experiments
if __name__== '__main__':
    save=bool(input('Insert any input to save'))
    experiments.run_cifar_user_input_bidi(save = save)

