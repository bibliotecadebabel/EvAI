
class ExperimentSettings():

    def __init__(self):

        self.init_dt_array = None
        self.max_init_iter = 0
        self.init_epochs = 0
        self.init_restart_period = 0
        self.init_dt_max = 0
        self.init_dt_min = 0

        self.joined_dt_array = None
        self.max_joined_iter = 0
        self.joined_epochs = 0
        self.joined_restart_period = 0
        self.joined_dt_max = 0
        self.joined_dt_min = 0

        self.best_dt_array = None
        self.max_best_iter = 0
        self.best_epochs = 0
        self.best_restart_period = 0
        self.best_dt_max = 0
        self.best_dt_min = 0

        self.period_save_space = 1
        self.period_save_model = 1
        self.period_new_space = 1

        self.batch_size = 128
        self.epochs = 100
        self.test_name = None

        self.weight_decay = None
        self.momentum = None
        self.cuda = True

        self.enable_activation = None
        self.enable_augmentation=None
        self.initial_dna = None
        self.initial_space = None

        self.dataGen = None

        self.selector = None
        self.allow_interupts = None
        self.enable_track_stats = None

        self.dropout_value = None
        self.dropout_function = None
        self.enable_last_activation = None
        self.version = None

        self.loadedNetwork = None
        self.save_txt = False
        self.disable_mutation = False
        self.eps_batchorm = None
        self.ricap = None
        self.evalLoss = False
        self.transformations_compose = None
