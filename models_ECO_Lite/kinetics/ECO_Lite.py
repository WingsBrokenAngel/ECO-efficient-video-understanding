# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-3-13

import tensorflow as tf
import tensornets as nets
import numpy as np
from tensorflow.layers import *




class EcoModel():
    def __init__(self, input_x, input_y):
        self.construct_model(input_x, input_y)
        self.initialize_model()
        self.input_x = input_x
        self.input_y = input_y
        self.cnn_trainable = False

    def construct_model(self, input_x, input_y):
        ct = self.cnn_trainable
        self.conv1_7x7_s2 = Conv2D(kernel_size=(7,7), filters=64, strides=2, padding='same', 
                            data_format='channels_first', trainable=ct, name='conv1_7x7_s2')

        self.conv1_7x7_s2_bn = BatchNormalization(axis=1, trainable=ct, name='conv1_7x7_s2_bn')

        self.pool1_3x3_s2 = MaxPooling2D(pool_size=3, strides=2, padding='valid', 
                                  data_format='channels_first', name='pool1_3x3_s2')
        
        self.conv2_3x3_reduce = Conv2D(kernel_size=1, filters=64, trainable=ct, 
                                       data_format='channels_first', name='conv2_3x3_reduce')
        
        self.conv2_3x3_reduce_bn = BatchNormalization(axis=1, trainable=ct, name='conv2_3x3_reduce_bn')
        
        self.conv2_3x3 = Conv2D(kernel_size=3, filters=192, padding='same', 
                                data_format='channels_first', trainable=ct, name='conv2_3x3')
        
        self.conv2_3x3_bn = BatchNormalization(axis=1, trainable=ct, name='conv2_3x3_bn')
        
        self.pool2_3x3_s2 = MaxPooling2D(pool_size=3, strides=2, padding='valid', 
                                         data_format='channels_first', name='pool2_3x3_s2')

        x = tf.reshape(input_x, (-1, 3, 224, 224))
        x = self.conv1_7x7_s2(x)
        x = self.conv1_7x7_s2_bn(x)
        x = tf.nn.relu(x, name='conv1_relu_7x7_inp')
        x = self.pool1_3x3_s2(x)

        x = self.conv2_3x3_reduce(x)
        x = self.conv2_3x3_reduce_bn(x)
        x = tf.nn.relu(x, name='conv2_relu_3x3_reduce_inp')

        x = self.conv2_3x3(x)
        x = self.conv2_3x3_bn(x)
        x = tf.nn.relu(x, name='conv2_relu_3x3_inp')

        x = self.pool2_3x3_s2(x, name='pool2_3x3_s2')

        x = self.inception_block_3a(x, ct)
        x = self.inception_block_3b(x, ct)




    def inception_block_3a(self, x, ct):
        self.inception_3a_1x1 = Conv2D(kernel_size=1, filters=64, data_format='channels_first', trainable=ct, 
                                       name='inception_3a_1x1')
        self.inception_3a_1x1_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3a_1x1_bn')
        self.inception_3a_3x3_reduce = Conv2D(kernel_size=1, filters=64, data_format='channels_first', trainable=ct,
                                              name='inception_3a_3x3_reduce')
        self.inception_3a_3x3_reduce_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3a_3x3_reduce_bn')
        self.inception_3a_3x3 = Conv2D(kernel_size=3, filters=64, padding='same', data_format='channels_first', 
                                       trainable='ct', name='inception_3a_3x3')
        self.inception_3a_3x3_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3a_3x3_bn')
        self.inception_3a_double_3x3_reduce = Conv2D(kernel_size=1, filters=64, data_format='channels_first', 
                                                     trainable=ct, name='inception_3a_double_3x3_reduce')
        self.inception_3a_double_3x3_reduce_bn = BatchNormalization(axis=1, trainable=ct, 
                                                                    name='inception_3a_double_3x3_reduce_bn')
        self.inception_3a_double_3x3_1 = Conv2D(kernel_size=3, filters=96, padding='same', 
                                                data_format='channels_first', trainable=ct, 
                                                name='inception_3a_double_3x3_1')
        self.inception_3a_double_3x3_1_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3a_double_3x3_1_bn')
        self.inception_3a_double_3x3_2 = Conv2D(kernel_size=3, filters=96, padding='same', 
                                                data_format='channels_first', trainable=ct, 
                                                name='inception_3a_double_3x3_2')
        self.inception_3a_double_3x3_2_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3a_double_3x3_2_bn')
        self.inception_3a_pool = MaxPooling2D(pool_size=3, strides=1, padding='same', data_format='channels_first', 
                                              name='inception_3a_pool')
        self.inception_3a_pool_proj = Conv2D(kernel_size=1, filters=32, data_format='channels_first', trainable=ct,
                                             name='inception_3a_pool_proj')
        self.inception_3a_pool_proj_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3a_pool_proj_bn')

        x1 = self.inception_3a_1x1(x)
        x1 = self.inception_3a_1x1_bn(x1)
        x1 = tf.nn.relu(x1)

        x2 = self.inception_3a_3x3_reduce(x)
        x2 = self.inception_3a_3x3_reduce_bn(x2)
        x2 = tf.nn.relu(x2)
        x2 = self.inception_3a_3x3(x2)
        x2 = self.inception_3a_3x3_bn(x2)
        x2 = tf.nn.relu(x2)

        x3 = self.inception_3a_double_3x3_reduce(x)
        x3 = self.inception_3a_double_3x3_reduce_bn(x3)
        x3 = tf.nn.relu(x3)
        x3 = self.inception_3a_double_3x3_1(x3)
        x3 = self.inception_3a_double_3x3_1_bn(x3)
        x3 = tf.nn.relu(x3)
        x3 = self.inception_3a_double_3x3_2(x3)
        x3 = self.inception_3a_double_3x3_2_bn(x3)
        x3 = tf.nn.relu(x3)

        x4 = self.inception_3a_pool(x)
        x4 = self.inception_3a_pool_proj(x4)
        x4 = self.inception_3a_pool_proj_bn(x4)
        x4 = tf.nn.relu(x4)
        
        x = tf.concat([x1, x2, x3, x4], axis=1, name='inception_3a_output')
        return x


    def inception_block_3b(self, x, ct):
        self.inception_3b_1x1 = Conv2D(kernel_size=1, filters=64, data_format='channels_first', trainable=ct, 
                                       name='inception_3b_1x1')
        self.inception_3b_1x1_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3b_1x1_bn')
        self.inception_3b_3x3_reduce = Conv2D(kernel_size=1, filters64, data_format='channels_first', trainable=ct,
                                              name='inception_3b_3x3_reduce')
        self.inception_3b_3x3_reduce_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3b_3x3_reduce_bn')
        self.inception_3b_3x3 = Conv2D(kernel_size=3, filters=96, padding='same', data_format='channels_first', 
                                       trainable=ct, name='inception_3b_3x3')
        self.inception_3b_3x3_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3b_3x3_bn')
        self.inception_3b_double_3x3_reduce = Conv2D(kernel_size=1, filters=64, data_format='channels_first', 
                                                     trainable=ct, name='inception_3b_double_3x3_reduce')
        self.inception_3b_double_3x3_reduce_bn = BatchNormalization(axis=1, trainable=ct, 
                                                                    name='inception_3b_double_3x3_reduce_bn')
        self.inception_3b_double_3x3_1 = Conv2D(kernel_size=3, filters=96, padding='same', data_format='channels_first', 
                                                trainable=ct, name='inception_3b_double_3x3_1')
        self.inception_3b_double_3x3_1_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3b_double_3x3_1_bn')
        self.inception_3b_double_3x3_2 = Conv2D(kernel_size=3, filters=96, padding='same', data_format='channels_first', 
                                                trainable=ct, name='inception_3b_double_3x3_2')
        self.inception_3b_double_3x3_2_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3b_double_3x3_2_bn')
        self.inception_3b_pool = MaxPooling2D(pool_size=3, strides=1, padding='same', data_format='channels_first', 
                                              name='inception_3b_pool')
        self.inception_3b_pool_proj = Conv2D(kernel_size=1, filters=64, data_format='channels_first', trainable=ct, 
                                             name='inception_3b_pool_proj')
        self.inception_3b_pool_proj_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3b_pool_proj_bn')

        x1 = self.inception_3b_1x1(x)
        x1 = self.inception_3b_1x1_bn(x1)
        x1 = tf.nn.relu(x1)

        x2 = self.inception_3b_3x3_reduce(x)
        x2 = self.inception_3b_3x3_reduce_bn(x2)
        x2 = tf.nn.relu(x2)
        x2 = self.inception_3b_3x3(x2)
        x2 = self.inception_3b_3x3_bn(x2)
        x2 = tf.nn.relu(x2)

        x3 = self.inception_3b_double_3x3_reduce(x)
        x3 = self.inception_3b_double_3x3_reduce_bn(x3)
        x3 = tf.nn.relu(x3)
        x3 = self.inception_3b_double_3x3_1(x3)
        x3 = self.inception_3b_double_3x3_1_bn(x3)
        x3 = tf.nn.relu(x3)
        x3 = self.inception_3b_double_3x3_2(x3)
        x3 = self.inception_3b_double_3x3_2_bn(x3)
        x3 = tf.nn.relu(x3)

        x4 = self.inception_3b_pool(x)
        x4 = self.inception_3b_pool_proj(x4)
        x4 = self.inception_3b_pool_proj_bn(x4)
        x4 = tf.nn.relu(x4)

        x = tf.concat([x1, x2, x3, x4], axis=1, name='inception_3b_output')
        return x


    def inception_block_3c(self, x, ct):
        self.inception_3c_double_3x3_reduce = Conv2D(kernel_size=1, filters=64, padding='valid', 
                                                     data_format='channels_first', trainable=ct, 
                                                     name='inception_3c_double_3x3_reduce')
        self.inception_3c_double_3x3_reduce_bn = BatchNormalization(axis=1, trainable=ct, 
                                                                    name='inception_3c_double_3x3_reduce_bn')
        self.inception_3c_double_3x3_1 = Conv2D(kernel_size=3, filters=96, padding='same', data_format='channels_first', 
                                                trainable=ct, name='inception_3c_double_3x3_1')
        self.inception_3c_double_3x3_1_bn = BatchNormalization(axis=1, trainable=ct, 
                                                               name='inception_3c_double_3x3_1_bn')

        x = self.inception_3c_double_3x3_reduce(x)
        x = self.inception_3c_double_3x3_reduce_bn(x)
        x = tf.nn.relu(x, name='inception_3c_relu_double_3x3_reduce_inp')

        x = self.inception_3c_double_3x3_1(x)
        x = self.inception_3c_double_3x3_1_bn(x)
        x = tf.nn.relu(x, name='inception_3c_relu_double_3x3_1_inp')
        x = tf.reshape(x, (-1, self.batch_size, 96, 28, 28), name='r2Dto3D')
        x = tf.transpose(x, [0, 2, 1, 3, 4])
        
