import os
import sys
sys.path.append(os.getcwd())

import subprocess
import shutil
from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument('--save_dir', default='videos', type=str)
parser.add_argument('--sample_index', nargs='+', default=[0, 10, 20], type=int)
parser.add_argument('--res_fn', required=True, type=str)
parser.add_argument('--wav_file', required=True, type=str)
args = parser.parse_args()

res_fn = args.res_fn
save_dir = args.save_dir
sample_index = args.sample_index
cur_wav_file = args.wav_file

with open(res_fn, 'r') as f:
    pred_res = np.array(json.load(f))

print('making video')
from visualise.draw_utils import *
pred_res = smooth(pred_res)

for index in sample_index:
    poses = cvt25(pred_res[index])
    output = draw_openpose_npy(720, 1280, 720, 1280, None, None, False, None, poses)
    image_dir = os.path.join(save_dir, 'sample_%d' % (index))
    save_images(output, image_dir)

    vid_file = os.path.join(save_dir, 'sample_%d.mp4' % (index))
    cmd = 'ffmpeg -y -i %s -i %s -r 25 -strict -2 %s' % (image_dir+'/%06d.jpg', cur_wav_file, vid_file)
    subprocess.call(cmd, shell=True)
    shutil.rmtree(image_dir)