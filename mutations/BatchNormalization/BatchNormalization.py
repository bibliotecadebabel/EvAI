
class MutateBatchNormalization():

    def __init__(self):
        pass

    def doMutate(self, oldBatchNorm, newLayer):

        if oldBatchNorm is not None and newLayer.getBatchNorm() is not None:

            new_batchNorm = newLayer.getBatchNorm()

            shape_oldBatch = oldBatchNorm.running_var.shape[0]
            shape_newBatch = new_batchNorm.running_var.shape[0]

            value = shape_newBatch - shape_oldBatch

            smallest_shape = shape_newBatch

            if value > 0:
                smallest_shape = shape_oldBatch
            elif value < 0:
                smallest_shape = shape_newBatch

            for i in range(smallest_shape):

                new_batchNorm.bias[i] = oldBatchNorm.bias[i].clone()
                new_batchNorm.weight[i] = oldBatchNorm.weight[i].clone()
                new_batchNorm.running_var[i] = oldBatchNorm.running_var[i].clone()
                new_batchNorm.running_mean[i] = oldBatchNorm.running_mean[i].clone()
            
            new_batchNorm.num_batches_tracked = oldBatchNorm.num_batches_tracked.clone()
            
            newLayer.setBarchNorm(new_batchNorm)

            del new_batchNorm

        