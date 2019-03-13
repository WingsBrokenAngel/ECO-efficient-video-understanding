# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-3-13

import tensorflow as tf
import tensornets as nets
import numpy as np
from tensorflow.layers import *
from tensorflow.losses import *
from tensorflow.nn import *
from tensorflow.summary import *
from pprint import pprint



class EcoModel():
    def __init__(self):
        self.cnn_trainable = False
        self.graph = tf.Graph()
        self.batch_size = 32
        with self.graph.as_default():
            self.input_x = tf.placeholder(dtype=tf.float32, shape=(None, 3, 224, 224))
            self.input_y = tf.placeholder(dtype=tf.int32, shape=(None,))
            self.construct_model(self.input_x, self.input_y)
            #self.initialize_model()


    def construct_model(self, input_x, input_y):
        ct = self.cnn_trainable
        x = self.inception_part(input_x, ct)
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))
        x = self.resnet_3d_part(x, ct)
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))
        x = AveragePooling3D(pool_size=(4, 7, 7), strides=(1, 1, 1), padding='valid', 
                             data_format='channels_first', name='global_pool')(x)
        x = tf.reshape(x, shape=(-1, 512))
        x = Dropout(0.3, name='dropout')(x)
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))
        self.fc8 = Dense(40, trainable=ct)
        x = self.fc8(x)
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))

        self.loss = sparse_softmax_cross_entropy(logits=x, labels=self.input_y)

        self.top1_acc = in_top_k(predictions=x, targets=self.input_y, k=1)
        self.top1_acc = tf.reduce_mean(tf.cast(self.top1_acc, tf.float32), name='top1_accuracy')
        self.top5_acc = in_top_k(predictions=x, targets=self.input_y, k=5)
        self.top5_acc = tf.reduce_mean(tf.cast(self.top5_acc, tf.float32), name='top5_accuracy')



    def inception_part(self, input_x, ct):
        self.conv1_7x7_s2 = Conv2D(kernel_size=(7,7), filters=64, strides=2, padding='same', 
                            data_format='channels_first', trainable=ct, name='conv1_7x7_s2')

        self.conv1_7x7_s2_bn = BatchNormalization(axis=1, trainable=ct, name='conv1_7x7_s2_bn')

        self.pool1_3x3_s2 = MaxPooling2D(pool_size=3, strides=2, padding='same', 
                                  data_format='channels_first', name='pool1_3x3_s2')
        
        self.conv2_3x3_reduce = Conv2D(kernel_size=1, filters=64, trainable=ct, 
                                       data_format='channels_first', name='conv2_3x3_reduce')
        
        self.conv2_3x3_reduce_bn = BatchNormalization(axis=1, trainable=ct, name='conv2_3x3_reduce_bn')
        
        self.conv2_3x3 = Conv2D(kernel_size=3, filters=192, padding='same', 
                                data_format='channels_first', trainable=ct, name='conv2_3x3')
        
        self.conv2_3x3_bn = BatchNormalization(axis=1, trainable=ct, name='conv2_3x3_bn')
        
        self.pool2_3x3_s2 = MaxPooling2D(pool_size=3, strides=2, padding='same', 
                                         data_format='channels_first', name='pool2_3x3_s2')

        x = tf.reshape(input_x, (-1, 3, 224, 224))
        x = self.conv1_7x7_s2(x)
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))

        x = self.conv1_7x7_s2_bn(x)
        x = tf.nn.relu(x, name='conv1_relu_7x7_inp')
        x = self.pool1_3x3_s2(x)
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))


        x = self.conv2_3x3_reduce(x)
        x = self.conv2_3x3_reduce_bn(x)
        x = tf.nn.relu(x, name='conv2_relu_3x3_reduce_inp')
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))

        x = self.conv2_3x3(x)
        x = self.conv2_3x3_bn(x)
        x = tf.nn.relu(x, name='conv2_relu_3x3_inp')
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))

        x = self.pool2_3x3_s2(x)
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))

        x = self.inception_block_3a(x, ct)
        x = self.inception_block_3b(x, ct)
        x = self.inception_block_3c(x, ct)
        return x




    def inception_block_3a(self, x, ct):
        self.inception_3a_1x1 = Conv2D(kernel_size=1, filters=64, data_format='channels_first', trainable=ct, 
                                       name='inception_3a_1x1')
        self.inception_3a_1x1_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3a_1x1_bn')
        self.inception_3a_3x3_reduce = Conv2D(kernel_size=1, filters=64, data_format='channels_first', trainable=ct,
                                              name='inception_3a_3x3_reduce')
        self.inception_3a_3x3_reduce_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3a_3x3_reduce_bn')
        self.inception_3a_3x3 = Conv2D(kernel_size=3, filters=64, padding='same', data_format='channels_first', 
                                       trainable=ct, name='inception_3a_3x3')
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
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))
        return x


    def inception_block_3b(self, x, ct):
        self.inception_3b_1x1 = Conv2D(kernel_size=1, filters=64, data_format='channels_first', trainable=ct, 
                                       name='inception_3b_1x1')
        self.inception_3b_1x1_bn = BatchNormalization(axis=1, trainable=ct, name='inception_3b_1x1_bn')
        self.inception_3b_3x3_reduce = Conv2D(kernel_size=1, filters=64, data_format='channels_first', trainable=ct,
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
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))
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
        print('{}: {}'.format(str(x), x.shape.as_list()[1:]))

        x = tf.nn.relu(x, name='inception_3c_relu_double_3x3_1_inp')
        x = tf.reshape(x, (-1, self.batch_size, 96, 28, 28), name='r2Dto3D')
        x = tf.transpose(x, [0, 2, 1, 3, 4])
        return x

    def resnet_3d_part(self, x, ct):
        x = self.res3_block(x, ct)
        x = self.res4_block(x, ct)
        x = self.res5_block(x, ct)
        return x


    def res3_block(self, x, ct):
        self.res3a_2n = Conv3D(kernel_size=3, filters=128, strides=1, padding='same', 
                               data_format='channels_first', trainable=ct, name='res3a_2n')
        self.res3a_bn = BatchNormalization(axis=1, trainable=ct, name='res3a_bn')
        self.res3b_1 = Conv3D(kernel_size=3, filters=128, strides=1, padding='same', 
                              data_format='channels_first', trainable=ct, name='res3b_1')
        self.res3b_1_bn = BatchNormalization(axis=1, trainable=ct, name='res3b_1_bn')
        self.res3b_2 = Conv3D(kernel_size=3, filters=128, strides=1, padding='same', 
                              data_format='channels_first', trainable=ct, name='res3b_2')
        self.res3b_2_bn = BatchNormalization(axis=1, trainable=ct, name='res3b_2_bn')
        
        x1 = self.res3a_2n(x)

        x2 = self.res3a_2n(x)
        x2 = self.res3a_bn(x2)
        x2 = tf.nn.relu(x2, name='res3a_relu')
        x2 = self.res3b_1(x2)
        x2 = self.res3b_1_bn(x2)
        x2 = tf.nn.relu(x2, name='res3b_1_relu')
        x2 = self.res3b_2(x2)

        x = x1 + x2
        x = self.res3b_2_bn(x)
        x = tf.nn.relu(x, name='res3b')
        return x

    def res4_block(self, x, ct):
        self.res4a_1 = Conv3D(kernel_size=3, filters=256, strides=2, padding='same', 
                              data_format='channels_first', trainable=ct, name='res4a_1')
        self.res4a_1_bn = BatchNormalization(axis=1, trainable=ct, name='res4a_1_bn')
        self.res4a_2 = Conv3D(kernel_size=3, filters=256, strides=1, padding='same', 
                              data_format='channels_first', trainable=ct, name='res4a_2')
        self.res4a_down = Conv3D(kernel_size=3, filters=256, strides=2, padding='same', 
                                 data_format='channels_first', trainable=ct, name='res4a_down')
        self.res4a_bn = BatchNormalization(axis=1, trainable=ct, name='res4a_bn')
        self.res4b_1 = Conv3D(kernel_size=3, filters=256, strides=1, padding='same', 
                              data_format='channels_first', trainable=ct, name='res4b_1')
        self.res4b_1_bn = BatchNormalization(axis=1, trainable=ct, name='res4b_1_bn')
        self.res4b_2 = Conv3D(kernel_size=3, filters=256, strides=1, padding='same', 
                              data_format='channels_first', trainable=ct, name='res4b_2')
        self.res4b_bn = BatchNormalization(axis=1, trainable=ct, name='res4b_bn')

        x1 = self.res4a_1(x)
        x1 = self.res4a_1_bn(x1)
        x1 = tf.nn.relu(x1, name='res4a_1_relu')
        x1 = self.res4a_2(x1)
        x2 = self.res4a_down(x)
        x = x1 + x2

        x1 = x
        x2 = self.res4a_bn(x)
        x2 = tf.nn.relu(x2, name='res4a_relu')
        x2 = self.res4b_1(x2)
        x2 = self.res4b_1_bn(x2)
        x2 = tf.nn.relu(x2, name='res4b_1_relu')
        x2 = self.res4b_2(x2)
        x = x1 + x2

        x = self.res4b_bn(x)
        x = tf.nn.relu(x, name='res4b_relu')

        return x

    def res5_block(self, x, ct):
        self.res5a_1 = Conv3D(kernel_size=3, filters=512, strides=2, padding='same', 
                              data_format='channels_first', trainable=ct, name='res5a_1')
        self.res5a_1_bn = BatchNormalization(axis=1, trainable=ct, name='res5a_1_bn')
        self.res5a_2 = Conv3D(kernel_size=3, filters=512, strides=1, padding='same', 
                              data_format='channels_first', trainable=ct, name='res5a_2')
        self.res5a_down = Conv3D(kernel_size=3, filters=512, strides=2, padding='same', 
                                 data_format='channels_first', trainable=ct, name='res5a_down')
        self.res5a_bn = BatchNormalization(axis=1, trainable=ct, name='res5a_bn')
        self.res5b_1 = Conv3D(kernel_size=3, filters=512, strides=1, padding='same', 
                              data_format='channels_first', trainable=ct, name='res5b_1')
        self.res5b_1_bn = BatchNormalization(axis=1, trainable=ct, name='res5b_1_bn')
        self.res5b_2 = Conv3D(kernel_size=3, filters=512, strides=1, padding='same', 
                              data_format='channels_first', trainable=ct, name='res5b_2')
        self.res5b_bn = BatchNormalization(axis=1, trainable=ct, name='res5b_bn')

        x1 = self.res5a_1(x)
        x1 = self.res5a_1_bn(x1)
        x1 = tf.nn.relu(x1, name='res5a_1_relu')
        x1 = self.res5a_2(x1)

        x2 = self.res5a_down(x)

        x = x1 + x2

        x1 = x
        x2 = self.res5a_bn(x)
        x2 = tf.nn.relu(x2, name='res5a_relu')
        x2 = self.res5b_1(x2)
        x2 = self.res5b_1_bn(x2)
        x2 = tf.nn.relu(x2, name='res5b_1_relu')
        x2 = self.res5b_2(x2)

        x = x1 + x2
        x = self.res5b_bn(x)
        x = tf.nn.relu(x, name='res5b_relu')
        return x


if __name__ == "__main__":
    model = EcoModel()
    with model.graph.as_default():
        # writer = FileWriter("./log/", tf.get_default_graph())
        # writer.close()
        collection_keys = model.graph.get_all_collection_keys()
