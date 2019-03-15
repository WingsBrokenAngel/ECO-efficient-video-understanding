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

    p = mp.Process(target=train_model_process, args=())
    plist.append(p)

    for p in plist:
        p.start()

    for p in plist:
        p.join()


def train_model_process(video_queue, epoch):
    model = EcoModel(cnn_trainable, batch_size)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    with tf.Session(graph=model.graph, config=config) as sess:
        # 加载数据
        full_iters = train_example_num // batch_size
        if train_example_num % batch_size:
            iter_num = full_iters + 1
        else:
            iter_num = full_iters

        for epoch_idx in range(epoch):
            for iter_idx in range(full_iters):
                batchx, batchy = np.zeros(shape=(batch_size, ))
                for batch_idx in range(batch_size):



        # 开始训练