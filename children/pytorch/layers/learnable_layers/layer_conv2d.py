import children.pytorch.layers.learnable_layers.layer_learnable as layer
import const.propagate_mode
from abc import abstractmethod
import torch

class Conv2dLayer(layer.LearnableLayer):

    def __init__(self, adn, torch_object, enable_activation, propagate_mode):
        
        layer.LearnableLayer.__init__(self, adn=adn, torch_object=torch_object)

        self.__enable_activation = enable_activation
        self.__batchnorm = None
        self.__pool = None
        self.__propagate_function = None

        self.__define_propagate_function(propagate_mode)

       
    def set_enable_activation(self, value):
        self.__enable_activation = value
    
    def set_pool(self, object_torch):
        self.__pool = object_torch
    
    def get_pool(self):
        return self.__pool

    def set_batch_norm_object(self, object_torch):
        self.__batchnorm = object_torch

    def get_batch_norm_object(self):
        return self.__batchnorm

    def set_batch_norm(self, value):

        if self.__batchnorm is not None and value is not None:
            self.__set_norm_bias(value.bias)
            self.__set_norm_weight(value.weight)
            self.__set_norm_var(value.running_var)
            self.__set_norm_mean(value.running_mean)
            self.__set_norm_batches_tracked(value.num_batches_tracked)

    def get_batch_norm(self):

        return self.__batchnorm

    def __set_norm_bias(self, value):
        self.__batchnorm.bias = torch.nn.Parameter(value.clone())
    
    def __set_norm_weight(self, value):
        self.__batchnorm.weight = torch.nn.Parameter(value.clone())
    
    def __set_norm_var(self, value):

        if value is not None:
            self.__batchnorm.running_var.data = value.data.clone()

    def __set_norm_mean(self, value):

        if value is not None:
            self.__batchnorm.running_mean.data = value.data.clone()
    
    def __set_norm_batches_tracked(self, value):

        if value is not None:
            self.__batchnorm.num_batches_tracked.data = value.data.clone()

    def __apply_normalization(self, tensor):

        norm = self.__batchnorm(tensor)
        return norm
    
    def __apply_pooling(self, tensor):
        
        value = self.__pool(tensor)
        return value

    def __define_propagate_function(self, propagate_mode):
        
        if propagate_mode == const.propagate_mode.CONV2D_PADDING:
            self.__propagate_function = self.__propagate_padding
        else:
            self.__propagate_function = self.__propagate_default

    def __propagate_default(self):

        current_input = self.__generate_input()

        if self.__pool is not None:
            current_input = self.__apply_pooling(current_input)

        if self.get_dropout_value() > 0:
            output_dropout = self.apply_dropout(current_input)
            value = self.object(output_dropout)
        else:
            value = self.object(current_input)

        value = self.__apply_normalization(value)
        
        self.value = value
        
        if self.__enable_activation == True:
            self.value = torch.nn.functional.relu(value)
    
    def __propagate_padding(self):

        current_input = self.__generate_input()

        if self.__pool is not None:
            current_input = self.__apply_pooling(current_input)

        kernel = self.adn[3]

        if self.node.kids[0].objects[0].adn[0] == 0:
            current_input = self.__do_pad_input(targetTensor=current_input, kernel_size=kernel)

        if self.get_dropout_value() > 0:
            output_dropout = self.apply_dropout(current_input)
            value = self.object(output_dropout)
        else:
            value = self.object(current_input)

        value = self.__apply_normalization(value)
        
        self.value = value
        
        if self.__enable_activation == True:
            self.value = torch.nn.functional.relu(value)

    def propagate(self):
        self.__propagate_function()
            
    def __generate_input(self):
    
        input_channels = self.adn[1]

        parents_outputs_channels = 0

        value = None
        
        for parent_layer in self.connected_layers:  
            parents_outputs_channels += parent_layer.value.shape[1]
        
        biggest_input_kernel = self.__get_input_with_bigger_kernel()

        if input_channels == parents_outputs_channels:

            concat_tensor_list = []

            for i in range(len(self.connected_layers)):
                
                current_input = self.connected_layers[i].value.clone()
                padded_input = self.__do_pad_kernels(current_input, biggest_input_kernel)
                del current_input
                concat_tensor_list.append(padded_input)

            value = torch.cat(tuple(concat_tensor_list), dim=1)

            for tensorPadded in concat_tensor_list:
                del tensorPadded
            
            del concat_tensor_list

        else:

            biggest_input_depth = self.__get_input_with_more_filters() 

            sum_tensor_list = []

            for i in range(len(self.connected_layers)):
                
                current_input = self.connected_layers[i].value.clone()
                padded_input_kernel = self.__do_pad_kernels(current_input, biggest_input_kernel)
                padded_input_depth = self.__do_pad_filters(padded_input_kernel, biggest_input_depth)
                del current_input, padded_input_kernel
                
                if i == 0:
                    padded_input_depth = padded_input_depth * self.tensor_h
                
                elif i == (len(self.connected_layers) - 1):
                    padded_input_depth = padded_input_depth * (1 - self.tensor_h)

                sum_tensor_list.append(padded_input_depth)

            value = sum_tensor_list[0]

            for tensor in sum_tensor_list[1:]:
                value += tensor

            for tensorPadded in sum_tensor_list:
                del tensorPadded
            
            del sum_tensor_list        

        return value
    
    def __get_input_with_bigger_kernel(self):

        kernel = 0
        biggest_input = None

        for layer in self.connected_layers:
            
            shape = layer.value.shape
            
            if kernel < shape[2]:

                kernel = shape[2]
                biggest_input = layer.value
        
        return biggest_input
    
    def __get_input_with_more_filters(self):

        depth = 0
        biggest_input = None

        for layer in self.connected_layers:
            
            shape = layer.value.shape
            
            if depth < shape[1]:

                depth = shape[1]
                biggest_input = layer.value
        
        return biggest_input
    
    def __do_pad_kernels(self, targetTensor, refferenceTensor):

        pad_tensor = targetTensor
        
        target_shape = targetTensor.shape
        refference_shape = refferenceTensor.shape

        diff_kernel = abs(refference_shape[2] - target_shape[2])

        if diff_kernel > 0:
            pad_tensor = torch.nn.functional.pad(targetTensor,(0, diff_kernel, 0, diff_kernel),"constant", 0)
        
        return pad_tensor

    def __do_pad_filters(self, targetTensor, refferenceTensor):

        pad_tensor = targetTensor
        
        target_shape = targetTensor.shape
        refference_shape = refferenceTensor.shape

        diff_depth = abs(refference_shape[1] - target_shape[1])

        if diff_depth > 0:
            pad_tensor = torch.nn.functional.pad(targetTensor,(0, 0, 0, 0, 0, diff_depth),"constant", 0)
        
        return pad_tensor
    
    def __do_pad_input(self, targetTensor, kernel_size):
        pad_tensor = targetTensor
        
        refference_size = kernel_size - 1

        if refference_size > 0:
            pad_tensor = torch.nn.functional.pad(targetTensor,(0, refference_size, 0, refference_size),"constant", 0)
        
        return pad_tensor
    
    def deleteParam(self):

        if self.object is not None:

            if hasattr(self.object, 'weight') and self.object.weight is not None:

                if hasattr(self.object.weight, 'grad') and self.object.weight.grad is not None:
                    del self.object.weight.grad

                del self.object.weight
            
            if hasattr(self.object, 'bias') and self.object.bias is not None:

                if hasattr(self.object.bias, 'grad') and self.object.bias.grad is not None:
                    del self.object.bias.grad

                del self.object.bias
            
            del self.object

            if self.__pool is not None:
                del self.__pool

            if self.__batchnorm is not None:

                if hasattr(self.__batchnorm, 'weight') and self.__batchnorm.weight is not None:
                    del self.__batchnorm.weight

                if  hasattr(self.__batchnorm, 'bias') and self.__batchnorm.bias is not None:
                    del self.__batchnorm.bias
                
                if hasattr(self.__batchnorm, 'running_var') and self.__batchnorm.running_var is not None:
                    del self.__batchnorm.running_var

                if hasattr(self.__batchnorm, 'running_mean') and self.__batchnorm.running_mean is not None:
                    del self.__batchnorm.running_mean
                
                if hasattr(self.__batchnorm, 'num_batches_tracked') and self.__batchnorm.num_batches_tracked is not None:
                    del self.__batchnorm.num_batches_tracked