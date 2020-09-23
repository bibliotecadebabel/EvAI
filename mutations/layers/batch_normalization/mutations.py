import children.pytorch.layers.learnable_layers.layer_conv2d as conv2d_layer
class MutateBatchNormalization():

    def __init__(self):
        pass

    def execute(self, oldBatchNorm, new_layer):

        if oldBatchNorm is not None and isinstance(new_layer, conv2d_layer.Conv2dLayer):

            new_batchNorm = new_layer.get_batch_norm()

            shape_oldBatch = oldBatchNorm.weight.shape[0]
            shape_newBatch = new_batchNorm.weight.shape[0]

            value = shape_newBatch - shape_oldBatch

            smallest_shape = shape_newBatch

            if value > 0:
                smallest_shape = shape_oldBatch
            elif value < 0:
                smallest_shape = shape_newBatch

            for i in range(smallest_shape):

                new_batchNorm.bias[i] = oldBatchNorm.bias[i].clone()
                new_batchNorm.weight[i] = oldBatchNorm.weight[i].clone()

                if oldBatchNorm.running_var is not None:
                    new_batchNorm.running_var[i] = oldBatchNorm.running_var[i].clone()
                
                if oldBatchNorm.running_mean is not None:
                    new_batchNorm.running_mean[i] = oldBatchNorm.running_mean[i].clone()
            
            if oldBatchNorm.num_batches_tracked is not None:
                new_batchNorm.num_batches_tracked = oldBatchNorm.num_batches_tracked.clone()
            
            new_layer.set_batch_norm(new_batchNorm)

            del new_batchNorm

        