# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:50:13 2018

@author: home
"""
import tensorflow as tf
from model import CGAN
import os
from image_utils import parse_single_example_for_image

def generate_data(tfrecords, parse_func, epochs, batch_size):
    "A easy implementation of Dataset API"
    "tfrecords - tfrecords file"
    "parse_func - parse the example in the tfrecords file"
    "batch_size - batch_size"
    Dataset = tf.data.TFRecordDataset(tfrecords) #decoposite the records into examples
    Dataset = Dataset.map(parse_func)
    Dataset = Dataset.shuffle(20)
    Dataset = Dataset.repeat(epochs)
    Dataset = Dataset.batch(batch_size)
    return Dataset
    
    
class CGAN_Model:
    
    def __init__(self, input_dim, output_dim, tfrecords, 
                 image_size = 286, 
                 crop_size  = 256,
                 batch_size = 4, 
                 epoches = 6,
                 GP_lambda = 10,
                 L1_lambda = 100, 
                 mode = "wgan_pg",
                 learning_rate = 2e-4,
                 save_iter = 1000,
                 save_path = "saver/cgan.ckpt"):
        
        self.input_dim  = input_dim
        self.output_dim = output_dim 
        
        self.input_tfrecords = tfrecords
        self.image_size = image_size
        self.crop_size  = crop_size
        self.batch_size = batch_size
        self.epoches   = epoches
        self.GP_lambda = GP_lambda
        self.L1_lambda = L1_lambda
        self.mode = mode
        self.lr = learning_rate
        self.save_iter = save_iter
        self.save_path = save_path
        
    def train(self):
        
        with tf.Graph().as_default(), tf.Session() as sess:
            
            with tf.device('/CPU:0'):
                #for content images
                #-------------------#
                parse_func = lambda x : parse_single_example_for_image(
                             x, image_size = self.image_size, crop_size = self.crop_size)
                Dataset = generate_data(self.input_tfrecords, parse_func, self.epoches, self.batch_size)
                iterator = Dataset.make_one_shot_iterator()#one shot generator
                next_element = iterator.get_next()#tensor size batch_size * crop_size * crop_size * 3
                inputs = tf.to_float(next_element)             
            
            
            self.cgan = CGAN(self.input_dim, self.output_dim, self.crop_size, self.batch_size, self.GP_lambda, self.L1_lambda, self.mode)    
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.cgan.d_loss, var_list=self.cgan.d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.cgan.g_loss, var_list=self.cgan.g_vars)
                          
            writer = tf.summary.FileWriter('C:/Users/home/Desktop/Tf_tutorial/logs', sess.graph)            
            
            # Initialize
            sess.run(tf.global_variables_initializer())
            g, _ = os.path.split(self.save_path)
            ckpt = tf.train.get_checkpoint_state(g)
            #load paramters
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(sess, ckpt.model_checkpoint_path)   
                
            saver = tf.train.Saver()
            iteration = 0
            
            try:
                while True: 
                    # for discriminator
                    fetches = [self.d_optim, self.cgan.d_sum]
                    feed_dict = {self.cgan.inputs: inputs.eval()}  
                    _, summaries = sess.run(fetches, feed_dict = feed_dict)
                    iteration += 1
                    writer.add_summary(summaries, iteration)
                    
                    #for generator
                    fetches = [self.g_optim, self.cgan.g_sum]
                    feed_dict = {self.cgan.inputs: inputs.eval()}  
                    _, summaries = sess.run(fetches, feed_dict = feed_dict)
                    iteration += 1
                    writer.add_summary(summaries, iteration)
                       
                    if iteration % self.save_iter == 0:
                        saver.save(sess, self.save_path, global_step = iteration)
                        
            except tf.errors.OutOfRangeError:
                #When training terminates, the parameters should be stored
                print('Traning terminates!')
                saver.save(sess, self.save_path, global_step = iteration)            
                
                  
        