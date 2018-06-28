# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 10:39:20 2018

@author: home
"""

import tensorflow as tf
from ops import conv_layer, Conv2D, UpSampling, linear


slim = tf.contrib.slim


batch_norm_parameters = {"running_mean"  : False,
                         "is_training"   : True,
                         "scale"         : True}
                         
def leaky_relu(nn, alpha = 0.2):
    return tf.nn.leaky_relu(nn, alpha)

class CGAN:
    
    
    def __init__(self, input_dim, output_dim, image_size = 256, batch_size = 4, GP_lambda = 0.1, L1_lambda = 100, mode = "wgan_gp"):
        
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.image_size = image_size
        self.GP_lambda = GP_lambda
        self.L1_lamdba = L1_lambda
        self.mode = mode
        self.build()
        
    def build(self):
        
        shape = [self.batch_size, self.image_size, self.image_size, self.input_dim+self.output_dim]
        self.inputs = tf.placeholder(tf.float32, shape = shape)
        self.real_A = self.inputs[:,:,:,:self.input_dim]
        self.real_B = self.inputs[:,:,:,self.input_dim:]
        self.fake_B = self.generator(self.real_A)
        
        self.real_data = tf.concat([self.real_A, self.real_B], axis = 3)
        self.fake_data = tf.concat([self.real_A, self.fake_B], axis = 3) 
        
        self.real_p, self.real_logits = self.discriminator(self.real_data)        
        self.fake_p, self.fake_logits = self.discriminator(self.fake_data, reuse = True)

        self.g_vars = slim.get_variables('generator')
        self.d_vars = slim.get_variables('discriminator')
        
        if self.mode == "cgan":
        
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.real_logits,
                                                                                      labels = tf.ones_like(self.real_p)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.fake_logits,
                                                                                      labels = tf.zeros_like(self.fake_p)))
            
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.fake_logits,
                                                                                 labels = tf.ones_like(self.fake_p))) + \
                          tf.reduce_mean(tf.abs(self.real_B - self.fake_B)) * self.L1_lamdba
    
            self.d_loss = self.d_loss_real + self.d_loss_fake
        
        elif self.mode == "wgan_gp":
            
            # Gradient penalty
            alpha = tf.random_uniform(
                shape=[self.batch_size, 1, 1, 1], 
                minval = 0.,
                maxval = 1.
            )
            
            differences = self.fake_data - self.real_data
            interpolates = self.real_data + (alpha * differences)
            gradients   = tf.gradients(self.discriminator(interpolates, reuse = True), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis = [1,2,3]))
         
            self.d_loss_real = - tf.reduce_mean(self.real_logits)
            self.d_loss_fake = tf.reduce_mean(self.fake_logits)
            self.d_loss = self.d_loss_real + self.d_loss_fake + tf.reduce_mean((slopes-1.)**2) * self.GP_lambda 
        
            self.g_loss = - tf.reduce_mean(self.fake_logits) + \
                            tf.reduce_mean(tf.abs(self.real_B - self.fake_B)) * self.L1_lamdba
    
        else:
            raise ValueError
        
        if self.output_dim + self.input_dim == 3:
            self.g_image = tf.summary.image("generated_image", self.convert_image(self.fake_data))
        elif self.output_dim == 3:
            self.g_image = tf.summary.image("generated_image",self.convert_image(self.fake_B))
            
        
        
        self.d_real_sum = tf.summary.histogram("real_p", self.real_p)
        self.d_fake_sum = tf.summary.histogram("fake_p", self.fake_p)        
        
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        
        self.g_sum = tf.summary.merge([self.g_image, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_fake_sum, self.d_real_sum, 
                                       self.d_loss_sum, self.d_loss_fake_sum, self.d_loss_real_sum])
        
      
    def convert_image(self, images):
        return (images + 1.0) / 2.0 * 255.0
        
    def generator(self, inputs, reuse = None):
        
        # used for skip connection in U-net
        tensor_records = []
        
        def Encoder_ops(nn, filters, kernel_size, stride, scope = None, tensor_records = tensor_records):
            nn = Conv2D(nn, filters, kernel_size, stride, scope)
            # store the tensor before activation function for skip connect            
            tensor_records.append(nn)
            nn = leaky_relu(nn)
            return nn
        
        def Decoder_ops(nn, filters, kernel_size, stride, scope = None, tensor_records = tensor_records):
            nn = UpSampling(nn, filters, kernel_size, stride, scope)
            # concat the tensor with the skip tensor before the activation funcion for symmetry
            nn = tf.concat([nn, tensor_records.pop()], axis = 3)
            nn = tf.nn.relu(nn)
            return nn
        
        nn = inputs #256
        
        with tf.variable_scope("generator", reuse = reuse):
            #the outer environment
            with slim.arg_scope([conv_layer], 
                                batch_norm = True, 
                                batch_norm_parameters = batch_norm_parameters,
                                activation = None,
                                weights_initializer = tf.truncated_normal_initializer(stddev = 0.02)):            
                #encoder
                with slim.arg_scope([conv_layer], batch_norm = False):
                    nn = Encoder_ops(nn, 64, 5, 2, 'conv1') #128
                nn = Encoder_ops(nn, 128, 5, 2, 'conv2')#64
                nn = Encoder_ops(nn, 256, 5, 2, 'conv3')#32
                nn = Encoder_ops(nn, 512, 5, 2, 'conv4')#16
                nn = Encoder_ops(nn, 512, 5, 2, 'conv5')#8
                nn = Encoder_ops(nn, 512, 5, 2, 'conv6')#4
                #nn = Encoder_ops(nn, 512, 4, 2, 'conv7')#2

                nn = leaky_relu(Conv2D(nn, 512, 5, 2, 'conv8')) #1
                #decoder    
                with slim.arg_scope([conv_layer], dropout = 0.5):

                    #nn = Decoder_ops(nn, 512, 4, 2, 'upconv1')#2
                    nn = Decoder_ops(nn, 512, 5, 2, 'upconv2')#4
                    nn = Decoder_ops(nn, 512, 5, 2, 'upconv3')#8  
                nn = Decoder_ops(nn, 512, 5, 2, 'upconv4')#16
                nn = Decoder_ops(nn, 256, 5, 2, 'upconv5')#32
                nn = Decoder_ops(nn, 128, 5, 2, 'upconv6')#64
                nn = Decoder_ops(nn, 64, 5, 2, 'upconv7') #128
                  
                #inverse back to the image domain    
                nn = tf.nn.tanh(UpSampling(nn, self.output_dim, 5, 2, 'upconv8')) #256
        return nn
                        
    def discriminator(self, inputs, reuse = None):
        
        nn = inputs
        batch_norm = True
        if self.mode == "wgan_gp":
            batch_norm = False

        with tf.variable_scope("discriminator", reuse = reuse):
            with slim.arg_scope([conv_layer], batch_norm = batch_norm, batch_norm_parameters = batch_norm_parameters, activation = leaky_relu): 
                
                with slim.arg_scope([conv_layer], batch_norm = False):
                    nn = Conv2D(nn, 64, 5, 2, 'conv1') #128
                
                nn = Conv2D(nn, 128, 5, 2, 'conv2') #64
                nn = Conv2D(nn, 256, 5, 2, 'conv3') #32
                nn = Conv2D(nn, 512, 5, 2, 'conv4') #16
                nn = linear(tf.reshape(nn, [self.batch_size, -1]), 1, 'linear1')

        return tf.nn.sigmoid(nn), nn                
                
                    