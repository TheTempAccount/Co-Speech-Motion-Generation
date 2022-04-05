import os
import sys

sys.path.append(os.getcwd())
from glob import glob

import numpy as np
import json

from nets import *
from repro_nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
import torch.nn as nn
from torch.utils import data
from scripts.infer import init_model

def get_norm_stats(args, config):
    if config.Data.pose.normalization:
        norm_stats_fn = os.path.join(os.path.dirname(args.model_path), "norm_stats.npy")
        norm_stats = np.load(norm_stats_fn, allow_pickle=True)
    else:
        norm_stats = None

    return norm_stats


def get_audio(args):
    audio_file = args.audio_file
    textgrid_file = args.textgrid_file

    return audio_file, textgrid_file

def load_initial_pose(args, norm_stats):
    initial_pose_file = args.initial_pose_file
    initial_poses = np.load(initial_pose_file)

    if norm_stats is not None:
        initial_poses = normalize(initial_poses, *norm_stats)

    initial_poses = torch.from_numpy(initial_poses)

    return initial_poses   


def save_res(wav_file, pred_res, exp_name):
    save_name = os.path.splitext(wav_file)[0] + '_%s.json'%(exp_name)
    with open(save_name, 'w') as f:
        json.dump(pred_res.tolist(), f)

def infer_for_one_speaker(data_root, speaker, generator, exp_name, infer_loader, infer_set, device, norm_stats, args=None):
    audio_text_pair = get_audio(data_root, speaker)
    for pair in audio_text_pair:
        cur_wav_file, cur_text_file = pair[0], pair[1]
        
        ite = iter(infer_loader)
        bat = next(ite)
        pre_poses = bat['pre_poses'].to(torch.float32).to(device)

        if args.same_initial:
            pre_poses = pre_poses[0:1].expand(pre_poses.shape[0], -1, -1)

        pred_res = generator.infer_on_audio(cur_wav_file,
            initial_pose = pre_poses,
            norm_stats = norm_stats,
            txgfile = cur_text_file 
        )
        save_res(cur_wav_file, pred_res, exp_name)

def main():
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)
    
    config = load_JsonConfig(args.config_file)

    print('init model...')
    generator = init_model(config.Model.model_name, args.model_path, args, config)
    
    norm_stats = get_norm_stats(args, config)
    cur_wav_file, cur_text_file = get_audio(args)
    initial_poses = load_initial_pose(args, norm_stats)

    pre_poses = initial_poses.to(torch.float32).to(device)
    if args.same_initial:
        pre_poses = pre_poses[0:1].expand(pre_poses.shape[0], -1, -1)

    pre_poses = pre_poses.permute(0, 2, 1)

    pred_res = generator.infer_on_audio(cur_wav_file,
            initial_pose = pre_poses,
            norm_stats = norm_stats,
            txgfile = cur_text_file 
        )
    save_res(cur_wav_file, pred_res, args.exp_name)

if __name__ == '__main__':
    main()