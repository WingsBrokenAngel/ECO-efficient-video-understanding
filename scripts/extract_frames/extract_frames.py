# -*- encoding: utf-8 -*-
'''
Author: 陈浩然
Date: 2018-10-30
'''

import numpy as np
import os
import subprocess
from pprint import pprint


def extract_frames():
    path1 = '/home/chenhaoran/YouTubeClips/'
    path2 = '/home/chenhaoran/YouTubeClips_frm/'
    frm_rate = '25'
    files = get_dir(path1)
    file_num = len(files)
    for i in range(file_num):
        cur_vid = os.path.join(path1, files[i])
        cur_dir = os.path.join(path2, files[i].strip().split('.')[0])
        try:
            os.mkdir(cur_dir)
        except FileExistsError as err:
            print('%s exists'%cur_dir)
        video_to_frame_command = ['ffmpeg', '-y', '-i', cur_vid, 
                                  '-vf', 'scale=%d:%d'%(224,224), 
                                  '-qscale:v', '2', '-r', frm_rate, 
                                  os.path.join(cur_dir, 'img_%4d.jpg')]
        msg = subprocess.run(video_to_frame_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pprint(msg)


def get_dir(path):
    '''
    获取目录下所有的视频文件名
    '''
    files = os.listdir(path)
    return files


if __name__ == "__main__":
    extract_frames()
