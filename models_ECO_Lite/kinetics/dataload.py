# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import multiprocessing as mp
import cv2
import os
import random
import time


class ImgArgument():
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.img = tf.placeholder(tf.float32, [256, 256, 3])
            img = tf.random_crop(self.img, [224, 224, 3])
            img = tf.image.random_flip_left_right(img)
            # img = tf.image.random_brightness(img, max_delta=32. / 255.)
            # img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
            img = tf.image.random_hue(img, max_delta=0.032)
            img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
            self.output = tf.clip_by_value(img, 0, 1.)


def load_one_video(path2video, num_frm, sample_size, is_train=True):
    '''
    Args:
    path2video: str, the absolute path to a video
    num_frm: int, the number of frames of that video
    sample_size: int, the number to sample
    is_train: bool, determine whether images should be argumented

    Return:
    a stack of frames for one video, shape = (sample_size, channels, height, width)
    Description: load a video
    '''
    frm_idx = np.random.choice(np.arange(num_frm), size=sample_size)
    frm_idx.sort()
    pic_names = os.listdir(path2video)
    images = []
    for idx in frm_idx:
        path2img = os.path.join(path2video, pic_names[idx])
        img = cv2.imread(path2img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = img / 255.

        if is_train:
            if random.randint(0, 1):
                img = cv2.flip(img, 1)
            x_start, y_start = random.randint(0, 32), random.randint(0, 32)
            img = img[x_start:x_start+224, y_start:y_start+224]
            delta = np.random.randn(224, 224, 3) * 0.05
            img += delta
        img = np.transpose(img, [2, 0, 1])
        img = (img - 0.5) * 2.0
        img = np.clip(img, -1., 1.)
        images.append(img)
    images = np.stack(images, axis=0)
    return images


def load_video(path2video_queue, video_queue, sample_size, is_train=True, idx=0):
    '''
    Args:
    path2video_queue: queue.Queue, the video paths are pushed into the queue
    video_queue: queue.Queue, the (np.array of video, label)s are pushed into the queue
    sample_size: int, the number of frames to be sampled
    is_train: bool, whether in training
    idx: int, the process index
    '''

    cnt = 0
    while True:
        try:
            path2video, num_frm, label = path2video_queue.get(True, 120)
            cnt += 1
        except:
            print('Process %d has finished. %d videos have been preprocessed.' % (idx, cnt))
            break

        images = load_one_video(path2video, num_frm, sample_size, is_train=is_train)
        video_queue.put([images, label])


def load_video_path(fpath, path2video_queue, epoch=1):
    warehouse = []
    with open(fpath, 'r') as fo:
        for line in fo:
            elems = line.strip().rsplit(' ', 2)
            elems = [e for e in elems if len(e) > 0]
            path2video, num_frm, label = elems[0], int(elems[1]), int(elems[2])
            warehouse.append((path2video, num_frm, label))


    for i in range(epoch):
        for elem in warehouse:
            path2video_queue.put(elem)
 
        random.shuffle(warehouse)


def get_images(video_queue):
    for i in range(3):
        images, label = video_queue.get()
        for j in range(images.shape[0]):
            img = images[j]
            img = np.transpose(img, [1, 2, 0])
            img = np.array(np.around((img / 2 + 0.5) * 255.), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('image_%d_%d_%d.jpg'%(i, j, label), img)

def test():
    path2train = '/home/chenhaoran/Kinetics400_rgb_train.txt'
    manager = mp.Manager()
    path2video_queue = manager.Queue()
    video_queue = manager.Queue()
    plist = []

    p = mp.Process(target=load_video_path, args=(path2train, path2video_queue, 1))
    plist.append(p)

    p = mp.Process(target=load_video, args=(path2video_queue, video_queue, 32, True, 0))
    plist.append(p)
    p = mp.Process(target=load_video, args=(path2video_queue, video_queue, 32, True, 1))
    plist.append(p)
    p = mp.Process(target=get_images, args=(video_queue, ))
    plist.append(p)

    for p in plist:
        p.start()

    time.sleep(60)
    print('video_queue size:', video_queue.qsize(), 'path2video_queue size:', path2video_queue.qsize())
    plist[0].terminate()
    plist[1].terminate()
    plist[2].terminate()
    plist[3].terminate()
    return

    for p in plist:
        p.join()



if __name__ == "__main__":
    test()