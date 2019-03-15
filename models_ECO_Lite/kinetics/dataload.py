# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import multiprocessing as mp
import cv2
import os


def load_one_video(path2video, num_frm, sample_size, sess, is_train=True):
    frm_idx = np.random.choice(np.arange(num_frm), size=sample_size, replacement=False)
    pic_names = os.listdir(path2video)
    images = []
    for idx in frm_idx:
        path2img = os.path.join(path2video, pic_names[idx])
        img = cv2.imread(path2img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = img / 255.

        crop_start_idx = np.random.randint(low=0, high=33, size=2)
        img_cropped = img[crop_start_idx[0]:crop_start_idx[0]+224, crop_start_idx[1]:crop_start_idx[1]+224]

        if np.random.randint(0, 2, size=1):
            img_cropped = cv2.flip(img_cropped, 1)

        if is_train:
            img = tf.random_crop(img, [224, 224, 3])
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=32. / 255.)
            img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
            img = tf.image.random_hue(img, max_delta=0.032)
            img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
            img = tf.clip_by_value(img, 0, 1.)
            img = sess.run(img)
        img = (img - 0.5) * 2.0 
        images.append(img)
    images = np.stack(images, axis=0)
    return images


def load_video(path2video_queue, video_queue, sample_size, is_train=True, idx=0):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    cnt = 0
    while True:
        try:
            path2video, num_frm, label = path2video_queue.get(True, 10)
            cnt += 1
        except:
            print('Process %d has finished. %d videos have been preprocessed.' % (idx, cnt))
            break

        images = load_one_video(path2video, num_frm, sample_size, sess, is_train=is_train)
        video_queue.put([images, label])



