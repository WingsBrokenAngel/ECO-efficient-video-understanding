# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Create Date: 2019-3-13
# Modification Date: 2019-3-14


import os
from scipy.io import loadmat
import argparse as ap
from pprint import pprint



action_map_path = 'class_ind_map_kinetics.mat'
frame_path = '/home/chenhaoran/kinetics-400-video'
save_path = '/home/chenhaoran/'


def create_list(action_map, split):
    path_trainList_rgb = os.path.join(save_path, 'YouTubeClips_rgb_%s.txt' % split)
    fileObj_trainList = open(path_trainList_rgb, 'w')
    split_path = os.path.join(frame_path, split + '_frm')
    dirList = os.listdir(split_path)
    for d in dirList:
        full_dir_path = os.path.join(split_path, d)
        vid_list = os.listdir(full_dir_path)
        label = action_map[d]
        for vid in vid_list:
            full_vid_path = os.path.join(full_dir_path, vid)
            vid_pics = os.listdir(full_vid_path)
            pic_num = len(vid_pics)
            fileObj_trainList.write('%s %d %d\n'%(full_vid_path, pic_num, label))
    fileObj_trainList.close()


def main():
    action_map = loadmat(action_map_path)['class_ind'].reshape((400,))
    amap = {}
    for elem in action_map:
        amap[elem[1][0]] = elem[0][0, 0]
    action_map = amap
    pprint(list(amap.items())[:10])
    
    create_list(action_map, 'train')    


if __name__ == "__main__":
    main()