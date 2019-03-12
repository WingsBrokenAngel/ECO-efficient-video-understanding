# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-3-12


import os
import subprocess


Path_Train = '/home/chenhaoran/kinetics-400-video/train'
Path_Val = '/home/chenhaoran/kinetics-400-video/val'
Path_Test = '/home/chenhaoran/kinetics-400-video/test'

Path_Train_Out = '/home/chenhaoran/kinetics-400-video/train_frm'
Path_Val_Out = '/home/chenhaoran/kinetics-400-video/val_frm'
Path_Test_Out = '/home/chenhaoran/kinetics-400-video/test_frm'


def get_all_dirs(path):
    dirs = os.listdir(path)
    full_dirs = [os.path.join(path, d) for d in dirs]
    return full_dirs


def iterate_vids(vids, out_path):
    for vid in vids:
        out_dir = os.path.join(out_path, os.path.basename(vid).split('.')[0])
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_dir_str = os.path.join(out_dir, 'img')

        cmd = ['ffmpeg', '-y', '-i', vid, '-vf', 'scale=%d:%d'%(320, 320), 
                '-qscale:v', '2', '-r', '4', out_dir_str + '%04d.jpg']

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def iterate_dirs(inpath, outpath):
    dirs = get_all_dirs(inpath)
    for idx, d in enumerate(dirs):
        vids = os.listdir(d)
        full_vids = [os.path.join(d, v) for v in vids]
        out_dir = os.path.join(outpath, os.path.basename(d))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        iterate_vids(full_vids, out_dir)
        print('\r%s   *** %5.2f%% have been finished.' % 
              (os.path.basename(inpath), (idx + 1.) / len(dirs)), end='')


def main():
    if not os.path.exists(Path_Train_Out):
        os.mkdir(Path_Train_Out)
    iterate_dirs(Path_Train, Path_Train_Out)

    if not os.path.exists(Path_Val_Out):
        os.mkdir(Path_Val_Out)

    iterate_dirs(Path_Val_Out, Path_Val_Out)

    if not os.path.exists(Path_Test_Out):
        os.mkdir(Path_Test_Out)

    iterate_dirs(Path_Test, Path_Test_Out)


if __name__ == "__main__":
    main()