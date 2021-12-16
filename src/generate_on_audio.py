from __future__ import absolute_import
from enum import Flag
from numpy.core.defchararray import not_equal

from torch.functional import norm

import math
import os
import random
import sys

from torch.serialization import save
from torch.utils import data
sys.path.append('..')

import time
import argparse
import numpy as np

from nets.Core import Core
from nets.discriminator import Discriminator
from nets.seq2seq import Seq2Seq
import torch
import torch.nn as nn
import logging
import json
import textgrid as tg
import numpy as np

import numpy as np
import python_speech_features
from scipy.io import wavfile
from scipy import signal
import os
from glob import glob
import json

device=torch.device('cpu')
# torch.cuda.set_device(device)

parser=argparse.ArgumentParser(description="Co-Speech Generation Model")
parser.add_argument('--model_name', required=True)
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--initial_pose', type=str, default='bill_initial.npy', help='the initial posture')
parser.add_argument('--audio_path', type=str, required=True)
parser.add_argument('--textgrid_path', type=str)
parser.add_argument('--visualize', action='store_false')
parser.add_argument('--sample_index', default=[0], type=int, nargs='+')

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--noise_size', default=256, type=int)
parser.add_argument('--normalization', action='store_true')
parser.add_argument('--audio_decoding', action='store_true')

parser.add_argument('--recon_input', action='store_true')
parser.add_argument('--augmentation', action='store_true')

parser.add_argument('--num_keypoints', default=54, type=int)
parser.add_argument('--hidden_size', default=1024, type=int)
parser.add_argument('--embed_size', default=1024, type=int)
parser.add_argument('--content_dim', default=512, type=int)
parser.add_argument('--enc_fc_layers', default=0, type=int)
parser.add_argument('--latent_fc_layers', default=1, type=int)
parser.add_argument('--dec_fc_layers', default=1, type=int)
parser.add_argument('--use_latent', default=1, type=int)
parser.add_argument('--use_prior', default=0, type=int)
parser.add_argument('--dec_interaction', default='add', type=str)
parser.add_argument('--T_layer_norm', default=1, type=int)
parser.add_argument('--dec_style', default=1, type=int)
parser.add_argument('--keypoint_weight', default=1, type=float)
parser.add_argument('--velocity_length', default=8, type=int)
parser.add_argument('--velocity_weight', default=5.0, type=float)
parser.add_argument('--velocity_start_weight', default=1e-5, type=float)
parser.add_argument('--velocity_decay_rate', default=0.99995, type=float)
parser.add_argument('--kl_weight', default=1.0, type=float)
parser.add_argument('--kl_start_weight', default=1e-5, type=float)
parser.add_argument('--kl_decay_rate', default=0.99995, type=float)
parser.add_argument('--kl_tolerance', default=0.01, type=float)
parser.add_argument('--cycle_model', default=1, type=int)
parser.add_argument('--cycle_weight', default=5.0, type=float)
parser.add_argument('--recon_input_weight', default=0.001, type=float)
parser.add_argument('--zero_loss_weight', default=1, type=int)

parser.add_argument('--learning_rate', default=0.0001, type=float, metavar='N')
parser.add_argument('--learning_rate_decay_factor', default=0.95, type=float, metavar='N')
parser.add_argument('--learning_rate_step',default=10000, type=int, metavar='N')
parser.add_argument('--max_gradient_norm', default=5, type=int, metavar='N', help="Clip gradients to this norm")
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='Batch size for each iteration')
parser.add_argument('--iterations', default=int(5e4), type=int, metavar='N', help="Iterations to train for")

parser.add_argument("--seq_length_in", default=25, type=int, metavar='N', help="Number of frames to feed into the encoder. 25 fps")
parser.add_argument("--seq_length_out", default=25, type=int, metavar='N', help="Number of frames that the decoder has to predict. 25fps")
parser.add_argument("--train_data", default=os.path.normpath("../pose_dataset/videos"), type=str, metavar='S', help="Data directory")
parser.add_argument("--test_data", default=os.path.normpath("../pose_dataset/videos"), type=str, metavar='S', help="Data directory")
parser.add_argument("--train_dir", default=os.path.normpath("./experiments/"), type=str, metavar='S', help="Training directory.")

parser.add_argument("--test_every", default=1000, type=int, metavar='N', help="How often to compute error on the test set.")
parser.add_argument("--save_every", default=1000, type=int, metavar='N', help="How often to compute error on the test set.")
parser.add_argument("--show_every", default=100, type=int, metavar='N', help="How often to show error during training.")
parser.add_argument("--use_cpu", action='store_true', help="Whether to use the CPU")

parser.add_argument('--load', type=str)
FLAGS = parser.parse_args()

print(FLAGS)

# FLAGS.audio_decoding = False
FLAGS.augmentation = False
# FLAGS.normalization = False

def denormalize(kps, data_mean, data_std):
    #kps: T, 108
    data_std = data_std.reshape(1, 1, -1)
    data_mean = data_mean.reshape(1, 1, -1)
    
    return (kps*data_std)+data_mean

def one_step_forward(last_step_results, code, audio_feat=None):
    inputs={
        'his_landmarks': last_step_results,
        'mfcc_feature': audio_feat
    }
    with torch.no_grad():
        if code == 1:
            outputs=generator.sample(inputs, 'trans')
        elif code == 0:
            outputs=generator.sample(inputs, data_type='zero')
        else:
            raise ValueError('code:%d'%(code))
    
    last_step_results=outputs['fut_landmarks']
    
    sample=outputs['fut_landmarks']
    sample=sample.cpu().detach().permute(1,0,2).numpy() #(B, seq, 50+2)
    if not FLAGS.normalization:
        sample*=1
    pred=sample.squeeze()
    pred_seq=pred[:,:,:108]
    if FLAGS.normalization:
        pred_seq = denormalize(pred_seq,data_mean, data_std)
    pred_seq=pred_seq.reshape(pred_seq.shape[0],pred_seq.shape[1], -1, 2) #(B, 25, 54, 2)
    # dummy_prob=np.ones((pred_seq.shape[0],25,54,1))
    pred_seq_concat=pred_seq
    pred_seq_concat=pred_seq_concat.reshape(pred_seq_concat.shape[0], pred_seq_concat.shape[1], -1)#(5,50)
    
    return last_step_results, pred_seq_concat

words=['but', 'as', 'to', 'that', 'with', 'of', 'the', 'and', 'or', 'not', 'which', 'what', 'this', 'for', 'because', 'if', 'so', 'just', 'about', 'like', 'by', 'how', 'from', 'whats', 'now', 'very', 'that', 'also', 'actually', 'who', 'then', 'well', 'where', 'even', 'today', 'between', 'than', 'when']
def parse_audio(textgrid_file, word_probs=None):
    txt=tg.TextGrid.fromFile(textgrid_file)
    # wp = json.load(open(word_probs))
    
    total_time=int(np.ceil(txt.maxTime))
    code_seq=np.zeros(total_time)
    
    word_level=txt[0]
    
    silen_period=[]
    
    for i in range(len(word_level)):
        start_time=word_level[i].minTime
        end_time=word_level[i].maxTime
        mark=word_level[i].mark
        
        if mark in words:
            start=int(np.round(start_time))
            end=int(np.round(end_time))
            
            if start >= len(code_seq) or end >= len(code_seq):
                code_seq[-1] = 1
            else:
                code_seq[start]=1

    return code_seq

def load_wav(fname, dest_sample=16000):
    sample_rate, sig = wavfile.read(fname)
    if sample_rate != dest_sample:
        result = int((sig.shape[0]) / sample_rate * dest_sample)
        x_resampled = signal.resample(sig, result)
        x_resampled = x_resampled.astype(np.float64)
        return dest_sample, x_resampled
    
    return sample_rate, sig

def extract_mfcc(audio,sample_rate=16000):
    mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
    mfcc = np.stack([np.array(i) for i in mfcc])
    return mfcc

def evaluate(init_pose, wav_file, textgrid_file=None):
    '''
    evaluate on a audio file
    '''
    all_res=[]
    res=[]

    if textgrid_file is not None:
        assert os.path.isfile(textgrid_file)
        code_seq=parse_audio(textgrid_file).astype(int)
    else:
        #if you do not have the textgrid file, you can specify the code seq by youself, make sure the length is no longer than the audio seconds and the value is 0 or 1. 
        code_seq = np.zeros(10)

    print(code_seq)
    sample_rate, sig = load_wav(wav_file)
    mfcc = extract_mfcc(sig)

    his_landmarks = np.load(init_pose, allow_pickle=True)

    if FLAGS.normalization:
        his_landmarks = (his_landmarks - data_mean) / data_std

    print(his_landmarks.shape)
    last_step_results=transform(his_landmarks) / 1 #*10

    for step in range(len(code_seq)):
        audio_feat = mfcc[:, step*100:step*100+100] #(13, 100)
        
        if audio_feat.shape[-1] != 100:
            audio_feat_pad = np.zeros((audio_feat.shape[0], 100))
            audio_feat_pad[:, :audio_feat.shape[-1]]=audio_feat
            audio_feat = audio_feat_pad
        
        audio_feat = audio_feat[np.newaxis,:].repeat(his_landmarks.shape[0], axis=0)
        audio_feat = torch.tensor(audio_feat, dtype = torch.float32).to(device)

        code=code_seq[step]
        last_step_results, pred_seq_concat=one_step_forward(last_step_results, code=code, audio_feat = audio_feat)
        res.append(pred_seq_concat)
        all_res.append(last_step_results.clone())
    
    pred_res=np.concatenate(res,axis=1)
    return pred_res, all_res

# get model
generator=Core(FLAGS, device).to(device)
model_name = FLAGS.model_name
model_path = FLAGS.model_path
init_pose = FLAGS.initial_pose

if FLAGS.normalization:
    normalize_stats = np.load(os.path.dirname(model_path)+'/norm_stats.npy')
    data_mean = normalize_stats[0]
    data_std = normalize_stats[1]

transform=lambda x: torch.tensor(x, dtype=torch.float32).permute(1,0,2).to(device)

best_checkpoint=torch.load(model_path, map_location=device)
generator.load_state_dict(best_checkpoint)
generator.eval()

# generation
cur_wav_file = FLAGS.audio_path
cur_tg_file = FLAGS.textgrid_path
print(cur_wav_file, cur_tg_file)

if not (os.path.isfile(cur_wav_file) and  os.path.isfile(cur_tg_file)):
    raise FileNotFoundError

pred_res, model_outputs = evaluate(init_pose, cur_wav_file, cur_tg_file)
pred_res = pred_res.tolist()

# save the generated results as json file
save_dir = os.path.join('results', model_name)
os.makedirs(save_dir, exist_ok=True)
save_name = os.path.splitext(os.path.basename(cur_wav_file))[0]+'_%s.json'%(model_name)
save_file = os.path.join(save_dir, save_name)
with open(save_file, 'w') as f:
    json.dump(pred_res, f)

if FLAGS.visualize:
    import subprocess
    import shutil
    print('making video')
    from visualize import *
    sample_index = FLAGS.sample_index
    pred_res = smooth(pred_res)

    #make a fake gt poses
    # his_landmarks = np.load(init_pose, allow_pickle=True)[0]#(64, 25, 108)
    # gt_poses = np.concatenate([his_landmarks, his_landmarks[::-1, :]], axis=0)#(50, 108)
    # seq_len = pred_res.shape[1]
    # while gt_poses.shape[0] < seq_len:
    #     gt_poses = np.concatenate([gt_poses, gt_poses], axis=0)
    # print(gt_poses.shape)

    for index in sample_index:
        poses = cvt25(pred_res[index])
        output = draw_openpose_npy(720, 1280, 720, 1280, None, None, False, None, poses)
        image_dir = os.path.join(save_dir, 'sample_%d' % (index))
        save_images(output, image_dir)

        vid_file = os.path.join(save_dir, 'sample_%d.mp4' % (index))
        cmd = 'ffmpeg -y -i %s -i %s -r 25 -strict -2 %s' % (image_dir+'/%06d.jpg', cur_wav_file, vid_file)
        subprocess.call(cmd, shell=True)
        shutil.rmtree(image_dir)