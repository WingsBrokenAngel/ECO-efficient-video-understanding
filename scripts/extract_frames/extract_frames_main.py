# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-3-12


import os
import subprocess
import multiprocessing as mp
import time



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
    num = mp.Value('i', 0)
    lock = mp.Lock()
    plist = []

    dirs = get_all_dirs(inpath)
    queue = mp.Queue()
    for d in dirs:
        queue.put(d)

    plist.append(mp.Process(target=print_info, args=(num, len(dirs), inpath)))

    for idx in range(16):
        p = mp.Process(target=single_process, args=(queue, lock, num, idx, outpath))
        plist.append(p)

    for p in plist:
        p.start()

    for p in plist:
        p.join()


def print_info(num, total_num, inpath):
    while num.value != total_num:
        time.sleep(60)
        print('\r%s   *** %5.2f%% have been finished.' % 
          (os.path.basename(inpath), num.value / total_num * 100.), end='')



def single_process(queue, lock, num, idx, outpath):
    while True:
        try:
            d = queue.get(True, 1)
        except:
            print('Process: %d' % idx, 'has finished')
            break

        vids = os.listdir(d)
        full_vids = [os.path.join(d, v) for v in vids]
        out_dir = os.path.join(outpath, os.path.basename(d))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        iterate_vids(full_vids, out_dir)
        
        lock.acquire()
        num.value += 1
        lock.release()



def main():
    # if not os.path.exists(Path_Train_Out):
    #     os.mkdir(Path_Train_Out)
    # iterate_dirs(Path_Train, Path_Train_Out)

    if not os.path.exists(Path_Val_Out):
        os.mkdir(Path_Val_Out)

    iterate_dirs(Path_Val, Path_Val_Out)

    # if not os.path.exists(Path_Test_Out):
    #     os.mkdir(Path_Test_Out)

    # iterate_dirs(Path_Test, Path_Test_Out)


if __name__ == "__main__":
    main()
