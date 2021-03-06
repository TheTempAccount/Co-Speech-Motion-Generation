import os
import sys
sys.path.append(os.getcwd())

import subprocess
import shutil
from argparse import ArgumentParser
import json
import numpy as np
from visualise.utils import get_gts, make_video, add_audio

parser = ArgumentParser()
parser.add_argument('--save_dir', default='videos', type=str)
parser.add_argument('--sample_index', nargs='+', default=[0], type=int)
parser.add_argument('--res_fn', required=True, type=str)
parser.add_argument('--wav_file', required=True, type=str)
parser.add_argument('--clip_pth', default=None, type=str)
args = parser.parse_args()

res_fn = args.res_fn
save_dir = args.save_dir
sample_index = args.sample_index
cur_wav_file = args.wav_file


# if clip_path given, use it to get face keypoints
if args.clip_pth is not None:
    _, gt_poses, _ = get_gts(args.clip_pth)
else:
    gt_poses = None

with open(res_fn, 'r') as f:
    pred_res = np.array(json.load(f))

print('making video')
from visualise.draw_utils import *
pred_res = smooth(pred_res)

for index in sample_index:
    poses = cvt25(pred_res[index], gt_poses)
    output = draw_openpose_npy(720, 1280, 720, 1280, None, None, False, None, poses)
    image_dir = os.path.join(save_dir, 'sample_%d' % (index))
    save_images(output, image_dir)

    vid_file = os.path.join(save_dir, 'sample_%d.mp4' % (index))
    # cmd = '~/anaconda3/envs/python2.7/bin/ffmpeg -y -i %s -i %s -r 25 -strict -2 %s' % (image_dir+'/%06d.jpg', cur_wav_file, vid_file)
    # subprocess.call(cmd, shell=True)
    make_video(image_dir, vid_file)
    shutil.rmtree(image_dir)
    add_audio(vid_file, cur_wav_file)