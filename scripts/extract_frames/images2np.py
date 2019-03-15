# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import glob


kinetics_path = '/home/chenhaoran/kinetics-400-video'
train_path = 'train_frm'
val_path = 'val_frm'


train_frm_full_path = os.path.join(kinetics_path, train_path)
val_frm_full_path = os.path.join(kinetics_path, val_path)

train_classes = os.listdir(train_frm_full_path)
train_full_path = [os.path.join(train_frm_full_path, c) for c in train_classes]

val_classes = os.listdir(val_frm_full_path)
val_full_path = [os.path.join(val_frm_full_path, v) for v in val_classes]

for path in train_full_path:
    vid_full_paths = glob.glob(os.path.join(path, '*'))
    for vid_path in vid_full_paths:
        img_full_paths = glob.glob(os.path.join(vid_path, '*'))
        if len(img_full_paths) < 12:
            continue
        imgs = []
        for img_full_path in img_full_paths:
            if img_full_path[-4:] == '.npy':
                continue
            img = cv2.imread(img_full_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        imgs = np.stack(imgs, axis=0)
        print(path, imgs.shape, imgs.dtype)
        np.save(os.path.join(vid_path, 'imgs'), imgs)


for path in val_full_path:
    vid_full_paths = glob.glob(os.path.join(path, '*'))
    for vid_path in vid_full_paths:
        img_full_paths = glob.glob(os.path.join(vid_path, '*'))
        if len(img_full_paths) < 12:
            continue
        imgs = []
        for img_full_path in img_full_paths:
            if img_full_path[-4:] == '.npy':
                continue
            img = cv2.imread(img_full_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        imgs = np.stack(imgs, axis=0)
        print(path, imgs.shape, imgs.dtype)
        np.save(os.path.join(vid_path, 'imgs'), imgs)