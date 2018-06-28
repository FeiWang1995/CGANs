# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:43:00 2018

@author: home
"""

import tensorflow as tf

slim = tf.contrib.slim


#sequential implementation
@slim.add_arg_scope
def conv_layer(inputs, 
               filters, 
               kernel_size, 
               stride,  
               padding = 'VALID',
               activation = tf.nn.relu,
               weights_initializer = tf.truncated_normal_initializer(stddev = 0.1),
               use_bias = True,
               bias_initializer = tf.constant_initializer(0.0),
               batch_norm = False,
               batch_norm_parameters = None,
               dropout = None,
               is_training = True,
               scope = None,
               reuse = None):

    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    
    if not params_shape.is_fully_defined():
        raise ValueError('Inputs %s has undefined last dimension %s.' % (inputs.name, params_shape))
    

    with tf.variable_scope(scope, 'conv', reuse = reuse):
        #for convolution
        shape = tf.TensorShape([kernel_size, kernel_size]).concatenate(params_shape).concatenate(tf.TensorShape([filters]))        
        weights = tf.get_variable("weights", 
                                  shape = shape,
                                  dtype = tf.float32,
                                  initializer = weights_initializer,
                                  trainable = True)
        outputs = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1], padding)
        
        #batch_norm op
        if batch_norm:
            eps = batch_norm_parameters.pop('epsilon', 1e-8)
            scale = batch_norm_parameters.pop('scale', True)
            decay = batch_norm_parameters.pop('decay', 0.9)
            trainable = batch_norm_parameters.pop('trainable', True)
            running_mean = batch_norm_parameters.pop('running_mean', False)
            
            if running_mean:
                update_collection = batch_norm_parameters.pop('update_collection', tf.GraphKeys.UPDATE_OPS)
                #moving mean and moving variance
                moving_mean = tf.get_variable('mean', [filters],
                                              initializer=tf.zeros_initializer,
                                              trainable=False)
                moving_variance = tf.get_variable('variance', [filters],
                                                  initializer=tf.zeros_initializer,
                                                  trainable=False)
                    
            if is_training: 
                def assign_moving_average(variable, value, decay):
                    return variable.assign(variable * decay + value * (1-decay))
                mean, variance = tf.nn.moments(outputs, axes = [0, 1, 2], name='moments')
                if running_mean:
                    tf.add_to_collection(update_collection, assign_moving_average(moving_mean, mean, decay))
                    tf.add_to_collection(update_collection, assign_moving_average(moving_variance, variance, decay))
                outputs -= mean
                outputs /= tf.sqrt(variance + eps)
            else:
                outputs -= moving_mean
                outputs /= tf.sqrt(moving_variance + eps)
               
            if scale:
                gamma = tf.get_variable("gamma",
                                        shape = [filters],
                                        dtype = tf.float32,
                                        initializer = tf.constant_initializer(1.0),
                                        trainable = trainable)
                beta = tf.get_variable("beta",
                                       shape = [filters],
                                       dtype = tf.float32,
                                       initializer = tf.constant_initializer(0.0),
                                       trainable = trainable)   
                outputs *= gamma
                outputs += beta

        else:
            if use_bias:
                bias = tf.get_variable("bias",
                                       shape = [filters],
                                       dtype = tf.float32,
                                       initializer = bias_initializer,
                                       trainable = True)
                outputs += bias
                
        if activation:
            outputs = activation(outputs)
            
        if dropout:
            outputs = tf.nn.dropout(outputs, dropout)
        
        return outputs
       
      
def Conv2D(inputs, filters, kernel_size, stride, scope = None):  

    #for padding in the paper
    #the kernel size is four
    assert stride in [1, 2], 'invalid stride!'
    assert kernel_size > 0, 'invalid kernel size!'
    
    if kernel_size % 2 == 0:
        padding   = (kernel_size // stride - 1) * stride
        padding_r = padding // 2
        padding_l = padding - padding_r
    else:
        padding = kernel_size // 2
        padding_r = padding
        padding_l = padding
        
    padded_input = tf.pad(
                   inputs, 
                   [[0, 0], [padding_l, padding_r], [padding_l, padding_r], [0, 0]], 
                   mode='REFLECT')
    
    output = conv_layer(padded_input, 
                        filters, 
                        kernel_size, 
                        stride,  
                        padding = 'VALID',
                        scope = scope)
    
    return output
    
def UpSampling(inputs, filters, kernel_size, stride = 2, scope = None):

    #upsampling with a convolution
    shape = inputs.get_shape().as_list()
    height = shape[1]
    width = shape[2]
    
    upsampled_input = tf.image.resize_nearest_neighbor(inputs, [stride * height, stride * width])
    #the stride is 1 for the same size as upsampled_input
    output = Conv2D(upsampled_input, filters, kernel_size, 1, scope)
    
    return output
    
    
def linear(inputs, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = inputs.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(inputs, matrix) + bias, matrix, bias
        else:
            return tf.matmul(inputs, matrix) + bias