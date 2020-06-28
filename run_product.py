import Experiments.product_experiments  as experiments
if __name__== '__main__':
    save=bool(input('Insert any input to save'))
    if input('Insert any input for remote'):
        experiments.run_cifar_user_input_bidi(save = save)
    else:
        experiments.run_local_ac()

#current remote experiment

#current local experiment
#experiments.run_cifar_user_input_no_save()
#experiments.run_local_no_image()

#experiments.run_cifar_user_input_bidi_save()
#experiments.run_cifar_user_no_image()
#experiments.run_local_no_image()
#experiments.run_local_ac()
##experiments.run_cifar_user_input_bidi()
#experiments.run_cifar_user_input_bidi_no_save()
#experiments.run_slow_ncf()
#experiments.run_cifar_user_input_save()
#experiments.run_cifar_user_input()
#import_test()
