from numpy.core.defchararray import upper
import pandas as pd
import numpy as np
import json
from glob import glob
from pandas.io.stata import PossiblePrecisionLoss
from python_speech_features.base import mfcc
import torch.utils.data as data
import os
import random
from .utils import *
from .checker import auto_label
from .augmentation import LimbScaling
import random
from tqdm import tqdm

class MultiVidData():
    def __init__(self, data_root, split='train', limbscaling=False, normalization=False):
        self.dataset={}
        self.sample_weight=[]
        self.complete_data=[]
        self.split = split
        self.limbscaling = limbscaling
        self.normalization = normalization
        self.data_root = data_root
        assert self.data_root.endswith('csv')
        self.df = pd.read_csv(self.data_root)
        self.df = self.df[self.df['dataset']==self.split]
        self.upper_body_points = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18] + list(np.arange(25, 25+21+21))

        
        self.generate_all_trans_frames_and_load_audio()

        self.trans_frames = np.array(self.trans_frames)
        self.zero_frames = np.array(self.zero_frames)
        print('num_trans:', len(self.trans_frames), 'num_zero:', len(self.zero_frames))
    
    def generate_all_trans_frames_and_load_audio(self):
        
        self.trans_frames = []
        self.zero_frames = []
        self.audio_lookup = {}
        
        for index in tqdm(range(len(self.df))):
            row = self.df.iloc[index]
            arr = row['pose_fn']
            poses = np.load(arr)['pose'].reshape(50, -1, 3) 
            assert poses.shape[1] == 25+21+21
            upper_body_points = poses[:, self.upper_body_points, :]
            confs = upper_body_points[..., -1:] 
            posepts = upper_body_points[..., 0:2]
            seq_data = posepts.reshape(50, -1)
            conf = np.concatenate([confs, confs], axis=-1) 
            conf = conf.reshape(50, -1)

            label = auto_label(seq_data, conf)
            if label  == 1:
                self.trans_frames.append(index)
            elif label == 0:
                self.zero_frames.append(index)
            else:
                
                pass

            audio_pth = (row['audio_fn'])
            if audio_pth in self.audio_lookup:
                pass
            else:
                _, rig = load_wav(audio_pth)
                mfcc = extract_mfcc(rig)
                self.audio_lookup[audio_pth] = mfcc

    def get_trans(self, source_seq_len, target_seq_len, batch_size):
        his_landmarks=[]
        fut_landmarks=[]
        his_confs = []
        fut_confs = []
        audio_features = []
        item_num=0
        
        
        indexes = random.sample(list(self.trans_frames), k=batch_size)
        for index in indexes:
            row = self.df.iloc[index]
            arr = (row['pose_fn'])
            start = row['start']
            end = row['end']
            audio_pth = (row['audio_fn'])
            poses = np.load(arr)['pose'].reshape(50, -1, 3) 

            upper_body_points = poses[:, self.upper_body_points, :]
            confs = upper_body_points[..., -1:] 
            posepts = upper_body_points[..., 0:2]
            seq_data = posepts.reshape(50, -1)
            conf = np.concatenate([confs, confs], axis=-1) 
            conf = conf.reshape(50, -1)

            mfcc_features = self.audio_lookup[audio_pth]
            start_sec = (start+25) / 25
            start_steps = int(start_sec / 0.01)
            mfcc_feature = mfcc_features[:, start_steps:start_steps+100]

            if mfcc_feature.shape[-1] < 100:
                mfcc_feature = np.pad(mfcc_feature, [0, 100 - mfcc_feature.shape[-1]], mode='constant', constant_values=0)

            his_landmarks.append(seq_data[:25, ...])
            fut_landmarks.append(seq_data[25:, ...])
            his_confs.append(conf[:25, ...])
            fut_confs.append(conf[25:, ...])
            audio_features.append(mfcc_feature)
        
        his_landmarks=np.array(his_landmarks)
        fut_landmarks=np.array(fut_landmarks)
        his_confs = np.array(his_confs)
        fut_confs = np.array(fut_confs)
        audio_features = np.array(audio_features)
        assert his_landmarks.shape[-1] == (12+21+21)*2

        if self.normalization:
            
            assert his_landmarks.shape[-1] == self.data_mean.shape[-1] and his_landmarks.shape[-1] == self.data_std.shape[-1]

            his_landmarks = (his_landmarks - self.data_mean) / self.data_std
            fut_landmarks = (fut_landmarks - self.data_mean) / self.data_std
        else:
            his_landmarks = his_landmarks / 1
            fut_landmarks = fut_landmarks / 1

        return his_landmarks, fut_landmarks, his_confs, fut_confs, audio_features
    
    def get_zero(self, source_seq_len, target_seq_len, batch_size):
        his_landmarks=[]
        fut_landmarks=[]
        his_confs = []
        fut_confs = []
        audio_features = []
        item_num=0
        indexes = random.sample(list(self.zero_frames), k=batch_size)
        for index in indexes:
            
            row = self.df.iloc[index]
            arr = (row['pose_fn'])
            start = row['start']
            end = row['end']
            audio_pth = (row['audio_fn'])
            poses = np.load(arr)['pose'].reshape(50, -1, 3) 

            upper_body_points = poses[:, self.upper_body_points, :]
            confs = upper_body_points[..., -1:] 
            posepts = upper_body_points[..., 0:2]
            seq_data = posepts.reshape(50, -1)
            conf = np.concatenate([confs, confs], axis=-1) 
            conf = conf.reshape(50, -1)

            mfcc_features = self.audio_lookup[audio_pth]
            start_sec = (start+25) / 25
            start_steps = int(start_sec / 0.01)
            mfcc_feature = mfcc_features[:, start_steps:start_steps+100]

            if mfcc_feature.shape[-1] < 100:
                mfcc_feature = np.pad(mfcc_feature, [0, 100 - mfcc_feature.shape[-1]], mode='constant', constant_values=0)

            his_landmarks.append(seq_data[:25, ...])
            fut_landmarks.append(seq_data[25:, ...])
            his_confs.append(conf[:25, ...])
            fut_confs.append(conf[25:, ...])
            audio_features.append(mfcc_feature)
        
        his_landmarks=np.array(his_landmarks)
        fut_landmarks=np.array(fut_landmarks)
        his_confs = np.array(his_confs)
        fut_confs = np.array(fut_confs)
        audio_features = np.array(audio_features)
        assert his_landmarks.shape[-1] == (12+21+21)*2

        if self.normalization:
            
            assert his_landmarks.shape[-1] == self.data_mean.shape[-1] and his_landmarks.shape[-1] == self.data_std.shape[-1]

            his_landmarks = (his_landmarks - self.data_mean) / self.data_std
            fut_landmarks = (fut_landmarks - self.data_mean) / self.data_std
        else:
            his_landmarks = his_landmarks / 1
            fut_landmarks = fut_landmarks / 1

        return his_landmarks, fut_landmarks, his_confs, fut_confs, audio_features

    def get_one(self, source_seq_len, target_seq_len, index):
        his_landmarks=[]
        fut_landmarks=[]
        his_confs = []
        fut_confs = []
        audio_features = []
        item_num=0
        
        
        row = self.df.iloc[index]
        arr = (row['pose_fn'])
        start = row['start']
        end = row['end']
        audio_pth = (row['audio_fn'])
        poses = np.load(arr)['pose'].reshape(50, -1, 3) 

        upper_body_points = poses[:, self.upper_body_points, :]
        confs = upper_body_points[..., -1:] 
        posepts = upper_body_points[..., 0:2]
        seq_data = posepts.reshape(50, -1)
        conf = np.concatenate([confs, confs], axis=-1) 
        conf = conf.reshape(50, -1)

        mfcc_features = self.audio_lookup[audio_pth]
        start_sec = (start+25) / 25
        start_steps = int(start_sec / 0.01)
        mfcc_feature = mfcc_features[:, start_steps:start_steps+100]

        if mfcc_feature.shape[-1] < 100:
            mfcc_feature = np.pad(mfcc_feature, [0, 100 - mfcc_feature.shape[-1]], mode='constant', constant_values=0)

        his_landmarks.append(seq_data[:25, ...])
        fut_landmarks.append(seq_data[25:, ...])
        his_confs.append(conf[:25, ...])
        fut_confs.append(conf[25:, ...])
        audio_features.append(mfcc_feature)
        
        his_landmarks=np.array(his_landmarks)
        fut_landmarks=np.array(fut_landmarks)
        his_confs = np.array(his_confs)
        fut_confs = np.array(fut_confs)
        audio_features = np.array(audio_features)
        assert his_landmarks.shape[-1] == (12+21+21)*2

        if self.normalization:
            
            assert his_landmarks.shape[-1] == self.data_mean.shape[-1] and his_landmarks.shape[-1] == self.data_std.shape[-1]

            his_landmarks = (his_landmarks - self.data_mean) / self.data_std
            fut_landmarks = (fut_landmarks - self.data_mean) / self.data_std
        else:
            his_landmarks = his_landmarks / 1
            fut_landmarks = fut_landmarks / 1

        return his_landmarks, fut_landmarks, his_confs, fut_confs, audio_features