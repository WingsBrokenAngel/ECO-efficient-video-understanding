# -*- coding: utf-8 -*-
import numpy as np
import os
import glob


dataset_path = '/home/chenhaoran/kinetics-400-video/train_frm'
save_path = '/home/chenhaoran/kinetics-400-npy'



def move_npy_files(dataset_path, save_path):
    classes_path = glob.glob(os.path.join(dataset_path, '*'))

    for class_path in classes_path:
        class_name = os.path.basename(class_path)
        class_save_path = os.path.join(save_path, class_name)
        if not os.path.exists(class_save_path):
            os.mkdir(class_save_path)

        vid_paths = glob.glob(os.path.join(class_path, '*'))
        for vid_path in vid_paths:
            vidname = os.path.basename(vid_path)
            npy_path = os.path.join(vid_path, 'imgs.npy')
            if os.path.exists(npy_path):
                vid_save_path = os.path.join(class_save_path, vidname) + '.npy'
                os.rename(npy_path, vid_save_path)


def main():
    move_npy_files('/home/chenhaoran/kinetics-400-video/train_frm', '/home/chenhaoran/kinetics-400-npy/train')
    move_npy_files('/home/chenhaoran/kinetics-400-video/val_frm', '/home/chenhaoran/kinetics-400-npy/val')