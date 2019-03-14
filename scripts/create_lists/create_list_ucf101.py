# -*- coding: utf-8 -*-
import os
import zipfile as zf
import json


def main():
    zip_path = '/home/chenhaoran/UCF101TrainTestSplits-RecognitionTask.zip'
    frame_dir = '/data/ucf101_rgb_img'

    myzip = zf.ZipFile(zip_path, 'r')
    namelist = myzip.namelist()[2:]

    action2id = json.load(open('/home/chenhaoran/UCF101_Action2Idx.json', 'r'))
    fail_cnt = 0
    for idx in range(len(namelist)):
        outpath = os.path.join('/home/chenhaoran/', os.path.basename(namelist[idx]))
        write_fo = open(outpath, 'w')
        fo = myzip.open(namelist[idx])
        lines = fo.readlines()
        lines = [str(line.strip().split()[0], encoding='utf-8') for line in lines]
        for line in lines:
            vid_frm_dir = os.path.join(frame_dir, line.split('.')[0])
            if os.path.exists(vid_frm_dir):
                frm_cnt = len(os.listdir(vid_frm_dir))
                label = action2id[line.split('/')[0]] - 1
                write_fo.write("%s %d %d\n" % (vid_frm_dir, frm_cnt, label))
            else:
                fail_cnt += 1

        fo.close()
        write_fo.close()

    print('Completed creating lists, failed: %d'% fail_cnt)


if __name__ == "__main__":
    main() 