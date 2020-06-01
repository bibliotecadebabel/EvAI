
class ExperimentSettings():

    def __init__(self):

        self.init_dt_array = None
        self.max_init_iter = 0

        self.joined_dt_array = None
        self.max_joined_iter = 0

        self.best_dt_array = None
        self.max_best_iter = 0

        self.period_save_space = 1
        self.period_save_model = 1
        self.period_new_space = 1

        self.batch_size = 128
        self.epochs = 100
        self.test_name = None

        self.weight_decay = 0.00001
        self.momentum = 0.9
        self.cuda = True

        self.enable_activation = True
        self.initial_dna = None
        self.initial_space = None

        self.dataGen = None

        self.selector = None
        self.allow_interupts = False
        self.enable_track_stats = True

        self.dropout_value = 0
        self.dropout_function = None
        self.enable_last_activation = True
        self.version = None
