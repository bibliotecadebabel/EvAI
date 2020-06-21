import time

class TestEntity():

    def __init__(self):
        self.id = 0
        self.name = None
        self.dt = 0
        self.dt_min = 0
        self.batch_size = 0
        self.max_layers = 0
        self.max_filters = 0
        self.max_filter_dense = 0
        self.max_kernel_dense = 0
        self.max_pool_layer = 0
        self.max_parents = 0
        self.start_time = 0
    
    def load(self, data):
        self.id = data[0]
        self.name = data[1]
        self.dt = data[2]
        self.dt_min = data[3]
        self.batch_size = data[4]
        self.max_layers = data[5]
        self.max_filters = data[6]
        self.max_filter_dense = data[7]
        self.max_kernel_dense = data[8]
        self.max_pool_layer = data[9]
        self.max_parents = data[10]
        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(data[11])))
