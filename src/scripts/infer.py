import os
import sys

from torch.utils import data
sys.path.append(os.getcwd())
from glob import glob

import torch
import torch.nn as nn
import numpy as np
import json

from nets import freeMo_Generator
from repro_nets import paper_Generator
from trainer.options import parse_args
from data_utils import torch_data

def init_model(model_name, model_path, args):
    if model_name == 'freeMo':
        generator = freeMo_Generator(args)
        generator.load_state_dict(torch.load(model_path)['generator'])
    elif model_name == 'freeMo_paper':
        generator = paper_Generator(args)
        model_ckpt = torch.load(model_path)
        if 'generator' in list(model_ckpt.keys()):
            generator.load_state_dict(model_ckpt['generator'])
        else:
            generator.load_state_dict(model_ckpt)
    else:
        raise NotImplementedError
    
    return generator

def init_dataloader(data_root, speakers, args):
    if data_root.endswith('.csv'):
        raise NotImplementedError
    else:
        data_class = torch_data

    data_base = torch_data(
        data_root=data_root,
        speakers=speakers,
        split='val',
        limbscaling=False,
        normalization=args.normalization,
        split_trans_zero=False,
        num_pre_frames=args.pre_pose_length,
        aud_feat_win_size=args.aud_feat_win_size,
        aud_feat_dim=args.aud_feat_dim,
        feat_method=args.feat_method
    )
    if args.normalization:
        norm_stats_fn = os.path.join(os.path.dirname(args.model_path), "norm_stats.npy")
        norm_stats = np.load(norm_stats_fn, allow_pickle=True)
        data_base.data_mean = norm_stats[0]
        data_base.data_std = norm_stats[1]
    else:
        norm_stats = None

    data_base.get_dataset()
    infer_set = data_base.all_dataset
    infer_loader = data.DataLoader(infer_set, batch_size=args.batch_size)

    return infer_set, infer_loader, norm_stats


def get_audio(data_root, speaker):
    
    audio_files = sorted(glob(os.path.join(data_root, "test_audios", speaker, "*.wav")))
    text_files = sorted(glob(os.path.join(data_root, "test_audios", speaker, "res", speaker, "*.TextGrid")))

    valid_audio = [os.path.splitext(os.path.basename(aud))[0] for aud in audio_files]
    valid_text = [os.path.splitext(os.path.basename(fn))[0] for fn in text_files]
    valid_samples = np.intersect1d(valid_audio, valid_text)

    audio_files = sorted([aud for aud in audio_files if os.path.splitext(os.path.basename(aud))[0] in valid_samples])
    text_files = sorted([fn for fn in text_files if os.path.splitext(os.path.basename(fn))[0] in valid_samples])

    audio_text_pair = zip(audio_files, text_files)

    return audio_text_pair

def save_res(wav_file, pred_res, exp_name):
    save_name = os.path.splitext(wav_file)[0] + '_%s.json'%(exp_name)
    with open(save_name, 'w') as f:
        json.dump(pred_res.tolist(), f)

def infer_for_one_speaker(data_root, speaker, generator, exp_name, infer_loader, infer_set, device, norm_stats):
    audio_text_pair = get_audio(data_root, speaker)
    for pair in audio_text_pair:
        cur_wav_file, cur_text_file = pair[0], pair[1]
        
        ite = iter(infer_loader)
        bat = next(ite)
        pre_poses = bat['pre_poses'].to(torch.float32).to(device)

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
    
    print('init model...')
    generator = init_model(args.model_name, args.model_path, args)
    print('init dataloader...')
    infer_set, infer_loader, norm_stats = init_dataloader(args.data_root, args.speakers, args)
    for speaker in args.speakers:
        
        infer_for_one_speaker(args.data_root, speaker, generator, args.exp_name, infer_loader, infer_set, device, norm_stats)

if __name__ == '__main__':
    main()