# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-3-15

import tensorflow as tf
import numpy as np
import os
from dataload import load_video, load_video_path
from ECO_Lite import EcoModel
import multiprocessing as mp


batch_size = 16
cnn_trainable = False
path2train = '/home/chenhaoran/Kinetics400_rgb_train.txt'
train_example_num = 229435
epoch = 10
frm_num = 32
beta = 0.0005# l2 loss parameter
display_interval = 100
gradient_bound = 40
learning_rate = 0.001

def train_model_global(train_video_path_file):
    # 设置训练环境
    manager = mp.Manager()
    path2video_queue = manager.Queue()
    video_queue = manager.Queue()
    plist = []
    p = mp.Process(targer=load_video_path, args=(path2train, path2video_queue, epoch))
    plist.append(p)
    for idx in range(8):
        p = mp.Process(target=load_video, args=(path2video_queue, video_queue, frm_num, True, idx))
        plist.append(p)

    model = EcoModel(cnn_trainable, batch_size)
    with model.graph.as_default():
        p = mp.Process(target=train_model_process, args=(model, video_queue, epoch))
        plist.append(p)

    for p in plist:
        p.start()

    for p in plist:
        p.join()


def train_model_process(model, video_queue, epoch):
    # Loss
    with tf.variable_scope('eco', reuse=tf.AUTO_REUSE):
        kernel_list = []
        kernel_list += [tf.get_variable('conv1_7x7_s2/kernel:0')]
        kernel_list += [tf.get_variable('conv2_3x3_reduce/kernel:0')]
        kernel_list += [tf.get_variable('conv2_3x3/kernel:0')]
        kernel_list += [tf.get_variable('inception_3a_1x1/kernel:0')]
        kernel_list += [tf.get_variable('inception_3a_3x3_reduce/kernel:0')]
        kernel_list += [tf.get_variable('inception_3a_3x3/kernel:0')]
        kernel_list += [tf.get_variable('inception_3a_double_3x3_reduce/kernel:0')]
        kernel_list += [tf.get_variable('inception_3a_double_3x3_1/kernel:0')]
        kernel_list += [tf.get_variable('inception_3a_double_3x3_2/kernel:0')]
        kernel_list += [tf.get_variable('inception_3a_pool_proj/kernel:0')]
        kernel_list += [tf.get_variable('inception_3b_1x1/kernel:0')]
        kernel_list += [tf.get_variable('inception_3b_3x3_reduce/kernel:0')]
        kernel_list += [tf.get_variable('inception_3b_3x3/kernel:0')]
        kernel_list += [tf.get_variable('inception_3b_double_3x3_reduce/kernel:0')]
        kernel_list += [tf.get_variable('inception_3b_double_3x3_1/kernel:0')]
        kernel_list += [tf.get_variable('inception_3b_double_3x3_2/kernel:0')]
        kernel_list += [tf.get_variable('inception_3b_pool_proj/kernel:0')]
        kernel_list += [tf.get_variable('inception_3c_double_3x3_reduce/kernel:0')]
        kernel_list += [tf.get_variable('inception_3c_double_3x3_1/kernel:0')]
        kernel_list += [tf.get_variable('res3a_2/kernel:0')]
        kernel_list += [tf.get_variable('res3b_1/kernel:0')]
        kernel_list += [tf.get_variable('res3b_2/kernel:0')]
        kernel_list += [tf.get_variable('res4a_1/kernel:0')]
        kernel_list += [tf.get_variable('res4a_2/kernel:0')]
        kernel_list += [tf.get_variable('res4a_down/kernel:0')]
        kernel_list += [tf.get_variable('res4b_1/kernel:0')]
        kernel_list += [tf.get_variable('res4b_2/kernel:0')]
        kernel_list += [tf.get_variable('res5a_1/kernel:0')]
        kernel_list += [tf.get_variable('res5a_2/kernel:0')]
        kernel_list += [tf.get_variable('res5a_down/kernel:0')]
        kernel_list += [tf.get_variable('res5b_1/kernel:0')]
        kernel_list += [tf.get_variable('res5b_2/kernel:0')]
        kernel_list += [tf.get_variable('fc8/kernel:0')]
    l2_loss_list = []
    for k in kernel_list:
        l2_loss_list.append(tf.nn.l2_loss(k))
    l2_loss = tf.reduce_sum(l2_loss_list)
    loss = model.loss + beta * l2_loss
    # Optimizer
    global_step = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(learning_rate, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads_vars = optimizer.compute_gradients(loss)
        grads_vars = [(tf.clip_by_norm(gv[0], gradient_bound), gv[1]) for gv in grads_vars]
        train_op = optimizer.apply_gradients(grads_vars)
    # Configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
    saver = tf.train.Saver()
    with tf.Session(graph=model.graph, config=config) as sess:
        # 加载数据
        sess.run(model.init_op)
        model.load_save(sess, '/home/chenhaoran/ECO-efficient-video-understanding/saves/init_model')
        full_iters = train_example_num // batch_size
        if train_example_num % batch_size:
            residue_examples = train_example_num % batch_size
        else:
            residue_examples = 0
        # 开始训练
        top1_list, top5_list = [], []
        for epoch_idx in range(epoch):
            for iter_idx in range(full_iters):
                batchx, batchy = np.zeros(shape=(batch_size, frm_num, 3, 224, 224), dtype=np.float32), \
                                            np.zeros(shape=(batch_size,), dtype=np.int32)
                for batch_idx in range(batch_size):
                    video_data, label = video_queue.get()
                    batchx[batch_idx, :, :, :, :] = video_data
                    batchy[batch_idx] = label

                res = sess.run([model.top1_acc, model.top5_acc, train_op], feed_dict={model.input_x: batchx, 
                                                                            model.input_y: batchy})
                top1_list.append(res[0])
                top5_list.append(res[1])
                if iter_idx % display_interval == display_interval - 1:
                    print('%5d epoch, %5d iter, top1 accuracy: %5.3f, top5 accuracy: %5.3f' % 
                          (epoch_idx, iter_idx, np.mean(top1_list), np.mean(top5_list)))
                    top1_list.clear()
                    top5_list.clear()

            if residue_examples:
                batchx, batchy = np.zeros(shape=(residue_examples, frm_num, 3, 224, 224), dtype=np.float32), \
                                                np.zeros(shape=(residue_examples, ), dtype=np.float32)
                for batch_idx in range(residue_examples):
                    video_data, label = video_queue.get()
                    batchx[batch_idx,:,:,:,:] = video_data
                    batchy[batch_idx] = label

                res = sess.run([model.top1_acc, model.top5_acc, train_op], feed_dict={model.input_x: batchx, 
                               model.input_y: batchy})
            saver.save(sess, '/home/chenhaoran/ECO-efficient-video-understanding/saves/model.ckpt', global_step)



