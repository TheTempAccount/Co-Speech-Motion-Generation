import os
import sys

sys.path.append(os.getcwd())

import subprocess
from glob import glob
import json

from visualise.draw_utils import draw_openpose_npy
from visualise.utils import load_json
from argparse import ArgumentParser
import numpy as np
import cv2 as cv

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--keypoint_dir', required=True, type=str)
    parser.add_argument('--aud_path', default=None, type=str)
    parser.add_argument('--save_dir', default='experiment/videos', type=str)
    parser.add_argument('--reference', required=True, type=str)
    parser.add_argument('--keep_img', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    keypoint_dir = args.keypoint_dir #should be dir/000001.json dir/000002.json
    aud_path = args.aud_path
    save_dir = args.save_dir
    reference = args.reference
    keep_img = args.keep_img

    jsons = sorted(glob(os.path.join(keypoint_dir, '*.json')))
    seq = []
    for fn in jsons:
        seq.append(load_json(fn))

    seq = np.array(seq)#(T, D, 3)

    crop_h = 720
    crop_w = 1280
    ims = draw_openpose_npy(crop_h, crop_w, crop_h, crop_w, None, None, False, None,  seq)

    im_save_dir = os.path.join(save_dir, reference, 'images')
    vid_save_name = os.path.join(save_dir, reference, 'video.avi')
    os.makedirs(im_save_dir, exist_ok=True)
    
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video_writer = cv.VideoWriter(vid_save_name, fourcc, 25, (1280, 720), True)

    for i in range(len(ims)):
        video_writer.write(ims[i])
        if keep_img:
            cv.imwrite(os.path.join(im_save_dir, '%06d.jpg'%(i)), ims[i])
    
    if aud_path is not None:
        cmd = '../anaconda3/envs/python2.7/bin/ffmpeg -i %s -i %s %s' % (vid_save_name, aud_path, os.path.join(save_dir, reference, 'video_audio.avi'))
        subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    main()
