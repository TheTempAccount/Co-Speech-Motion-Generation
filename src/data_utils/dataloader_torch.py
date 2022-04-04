import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import json
from glob import glob
from data_utils.utils import *
from data_utils.checker import auto_label
from data_utils.augmentation import LimbScaling
import torch.utils.data as data
from data_utils.consts import speaker_id


class PoseDataset():
    '''
    creat a dataset for every segment and concat. 
    '''
    def __init__(self, 
                data_root,
                speaker,
                
                audio_fn, 
                audio_sr,
                fps,
                feat_method='mel_spec',
                audio_feat_dim=64,
                audio_feat_win_size=None,

                train=True,
                load_all=False,
                split_trans_zero = False, 
                limbscaling=False, 
                num_frames=25, 
                num_pre_frames=25,
                context_info=False
                ):
        
        self.data_root=data_root
        self.speaker = speaker

        self.feat_method = feat_method
        self.audio_fn = audio_fn
        self.audio_sr = audio_sr
        self.fps = fps
        self.audio_feat_dim = audio_feat_dim
        self.audio_feat_win_size = audio_feat_win_size
        self.context_info = context_info #for aud feat

        self.train=train
        self.load_all = load_all
        self.split_trans_zero = split_trans_zero
        self.limbscaling = limbscaling
        self.num_frames = num_frames
        self.num_pre_frames = num_pre_frames
        
        self.annotations=glob(data_root+'/keypoints_new/person_1/*json')
        if len(self.annotations)==0:
            raise FileNotFoundError(data_root+'/keypoints are empty')
        self.annotations=sorted(self.annotations)
        self.img_name_list = self.annotations

        if self.load_all:
            self._load_them_all(split_trans_zero=self.split_trans_zero)
        
        if self.split_trans_zero:
            self.trans_frames=np.array(self.trans_frames)
            self.zero_frames=np.array(self.zero_frames)

            intesection = [i for i in self.zero_frames if i in self.trans_frames]
            assert len(intesection) == 0
        
    def _load_them_all(self, split_trans_zero=False):
        self.loaded_data={}
        self.conf={}
        self.complete_data=[]
        upper_body_points = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]
        
        neck_to_nose_len = []
        mean_position = []
        for img_name_ori in self.img_name_list:
            img_name=os.path.splitext(os.path.basename(img_name_ori))[0]
            annotation_file=os.path.join(self.data_root, 'keypoints_new/person_1/', img_name+'.json')
            if not os.path.isfile(annotation_file):
                continue
            annotation=json.load(open(annotation_file))['people'][0]

            posepts=annotation['pose_keypoints_2d']
            hand_left_pts=annotation['hand_left_keypoints_2d']
            hand_right_pts=annotation['hand_right_keypoints_2d']

            neck = np.array(posepts).reshape(-1,3)[1]
            nose = np.array(posepts).reshape(-1,3)[0]
            x_offset = abs(neck[0]-nose[0])
            y_offset = abs(neck[1]-nose[1])
            neck_to_nose_len.append(y_offset)
            mean_position.append([neck[0],neck[1]])
            
            
            keypoints=np.array(posepts+hand_left_pts+hand_right_pts).reshape(-1,3)[:,:2]
            conf=np.array(posepts+hand_left_pts+hand_right_pts).reshape(-1,3)[:,2:]

            upper_body = keypoints[upper_body_points, :]
            hand_points = keypoints[25:, :]
            keypoints = np.vstack([upper_body, hand_points])

            upper_body_conf = conf[upper_body_points, :]
            hand_points_conf = conf[25:, :]
            conf = np.vstack([upper_body_conf, hand_points_conf])
            conf=np.hstack([conf, conf])
            assert (keypoints.shape[0]==12+21+21) and (keypoints.shape[1]==2)
            assert (conf.shape[0]==12+21+21) and (conf.shape[1]==2)
            
            self.conf[img_name] = conf.reshape(-1)
            self.loaded_data[img_name]=keypoints.reshape(-1)
    
        
        if len(neck_to_nose_len) > 0:
            scale_factor = np.mean(neck_to_nose_len)
        else:
            print('no frame with angle less than 0.1:')
            raise ValueError

        mean_position=np.mean(np.array(mean_position), axis=0)
        self.localize_stats = (scale_factor, mean_position)

        for img_name in self.img_name_list:
            img_name=os.path.splitext(os.path.basename(img_name))[0]
            keypoints = np.array(self.loaded_data[img_name]).reshape(-1,2)
            assert keypoints.shape[0] == 12+21+21
            neck = keypoints[1].copy()

            keypoints[:, 0] = (keypoints[:, 0] - neck[0]) / scale_factor 
            keypoints[:, 1] = (keypoints[:, 1] - neck[1]) / scale_factor
            self.loaded_data[img_name] = keypoints.reshape(-1)
            self.complete_data.append(keypoints.reshape(-1))

        self.complete_data=np.array(self.complete_data)

        if self.audio_feat_win_size is not None:
            self.audio_feat = get_mfcc_old(self.audio_fn).transpose(1, 0)
            # print(self.audio_feat.shape)
        else:
            if self.feat_method == 'mel_spec':
                self.audio_feat = get_melspec(self.audio_fn, fps = self.fps, sr = self.audio_sr, n_mels=self.audio_feat_dim)
            elif self.feat_method == 'mfcc':
                self.audio_feat = get_mfcc_psf(self.audio_fn, 
                                            fps = self.fps, 
                                            sr = self.audio_sr, 
                                            n_mfcc=self.audio_feat_dim, 
                                            win_size=self.audio_feat_win_size
                                        )

        if split_trans_zero:
            self.generate_all_trans_frames()

    def generate_all_trans_frames(self):
        
        self.trans_frames = []
        self.zero_frames = []
        seq_len = self.num_frames+self.num_pre_frames
        for index in range(0, len(self)-seq_len-1):
            seq_data=[]
            conf=[]
            for i in range(index, index+seq_len):
                img_name=self.img_name_list[i]
                img_name=os.path.splitext(os.path.basename(img_name))[0]
                pose_at_i=self.loaded_data[img_name]
                conf_at_i=self.conf[img_name]
                conf.append(conf_at_i)
                seq_data.append(pose_at_i)
            seq_data=np.array(seq_data)
            conf=np.array(conf)

            label = auto_label(seq_data, conf, speaker=self.speaker)
            if label  == 1:
                self.trans_frames.append(index)
            elif label == 0:
                self.zero_frames.append(index)
            else:
                pass

    def get_dataset(self, normalization=False, normalize_stats=None):
        
        class __Worker__(data.Dataset):
            def __init__(child, index_list, normalization, normalize_stats, name='trans') -> None:
                super().__init__()
                child.index_list = index_list
                child.normalization = normalization
                child.normalize_stats = normalize_stats
                child.name = name

            def __getitem__(child, index):
                
                
                num_frames = self.num_frames
                num_pre_frames = self.num_pre_frames

                seq_len=num_frames+num_pre_frames
                assert index+seq_len<len(self)
                seq_data=[]
                conf=[]
                index = child.index_list[index]
                for i in range(index, index+seq_len):
                    img_name=self.img_name_list[i]
                    img_name=os.path.splitext(os.path.basename(img_name))[0]
                    pose_at_i=self.loaded_data[img_name]
                    conf_at_i=self.conf[img_name]
                    conf.append(conf_at_i)
                    seq_data.append(pose_at_i)
                seq_data=np.array(seq_data)

                if self.limbscaling and child.name == 'trans':
                    if np.random.random() < 0.3:
                        global_scale = 1
                        local_scales = 0.75 + np.random.random([12])*(1.25-0.75)
                        seq_data = LimbScaling(seq_data, global_scale, local_scales)

                conf=np.array(conf)
                pre_poses = (seq_data[:num_pre_frames, :])  
                pre_conf = conf[:num_pre_frames, :]
                poses = (seq_data[num_pre_frames:, :])
                conf = conf[num_pre_frames:, :]

                '''
                audio feature，
                context info: history and future
                '''
                if not self.context_info:
                    if self.audio_feat_win_size is None:
                        audio_feat = self.audio_feat[index+num_pre_frames:index+seq_len, ...]
                        if audio_feat.shape[0] < self.num_frames:
                            audio_feat = np.pad(audio_feat, [[0, self.num_frames-audio_feat.shape[0]], [0, 0]], mode='reflect')
                        
                        assert audio_feat.shape[0] == self.num_frames and audio_feat.shape[1] == self.audio_feat_dim
                    else:
                        #older version, where audio_feat_win_size is 100/1s
                        #TODO: not ready
                        start_sec = (index + num_frames) / 25 
                        start_steps = int(start_sec / 0.01) 
                        # print(self.audio_feat.shape, start_steps)
                        mfcc_feature = self.audio_feat[start_steps:start_steps+100, :]
                        # print(mfcc_feature.shape)
                        if mfcc_feature.shape[0] != 100:
                            mfcc_feature = np.pad(mfcc_feature, [[0, 100 - mfcc_feature.shape[-1]], [0, 0]], mode='reflect')
                        audio_feat = mfcc_feature
                else:#including feature and history
                    if self.audio_feat_win_size is None:
                        audio_feat = self.audio_feat[index:index+seq_len+num_frames, ...]
                        if audio_feat.shape[0] < seq_len+num_frames:
                            audio_feat = np.pad(audio_feat, [[0, seq_len + self.num_frames-audio_feat.shape[0]], [0, 0]], mode='constant')
                        
                        assert audio_feat.shape[0] == self.num_frames+seq_len and audio_feat.shape[1] == self.audio_feat_dim
                    else:
                        #older version, where audio_feat_win_size is 100/1s
                        #TODO: not ready
                        raise NotImplementedError
                        start_sec = (index) / 25 
                        start_steps = int(start_sec / 0.01) 
                        mfcc_feature = self.audio_feat[start_steps:start_steps+100, :]
                        if mfcc_feature.shape[0] != 100:
                            mfcc_feature = np.pad(mfcc_feature, [[0, 100 - mfcc_feature.shape[-1]], [0, 0]], mode='constant')
                        audio_feat = mfcc_feature

                if child.normalization:
                    data_mean = child.normalize_stats['mean'].reshape(1, 108)
                    data_std = child.normalize_stats['std'].reshape(1, 108)

                    pre_poses = (pre_poses - data_mean) / data_std
                    poses = (poses - data_mean) / data_std

                data_sample={
                    'pre_poses': pre_poses.astype(np.float).transpose(1, 0),
                    'pre_conf': pre_conf.astype(np.float).transpose(1, 0),
                    'poses': poses.astype(np.float).transpose(1, 0),
                    'conf': conf.astype(np.float).transpose(1, 0),
                    'aud_feat': audio_feat.astype(np.float).transpose(1, 0),
                    'speaker': speaker_id[self.speaker]
                }
                return data_sample

            def __len__(child):
                return len(child.index_list)

        if self.split_trans_zero:
            trans_index_list=self.trans_frames[self.trans_frames<(len(self)-self.num_frames-self.num_pre_frames-1)]
            zero_index_list=self.zero_frames[self.zero_frames<(len(self)-self.num_pre_frames-self.num_frames-1)]

            if len(trans_index_list) > 0:
                self.trans_dataset = __Worker__(trans_index_list, normalization, normalize_stats, name='trans')
            else:
                self.trans_dataset = None
        
            if len(zero_index_list) > 0:
                self.zero_dataset = __Worker__(zero_index_list, normalization, normalize_stats, name='zero')
            else:
                self.zero_dataset = None
        else:
            index_list = list(range(0, len(self)-self.num_frames-self.num_pre_frames-1))
            self.all_dataset = __Worker__(index_list, normalization, normalize_stats, name='all')

    def __len__(self):
        return len(self.img_name_list)

class MultiVidData():
    def __init__(self, 
                data_root, 
                speakers, 
                split='train', 
                limbscaling=False, 
                normalization=False,
                norm_method='new',
                split_trans_zero=False,
                num_frames=25,
                num_pre_frames=25,
                aud_feat_win_size=None,
                aud_feat_dim=64,
                feat_method='mel_spec',
                context_info=False
                ):
        self.data_root = data_root
        self.speakers = speakers
        self.split = split
        self.norm_method=norm_method
        self.normalization = normalization
        self.limbscaling = limbscaling
        self.num_frames=num_frames
        self.num_pre_frames=num_pre_frames
        self.split_trans_zero=split_trans_zero
        
        if self.split_trans_zero:
            self.trans_dataset_list = []
            self.zero_dataset_list = []
        else:
            self.all_dataset_list = []
        self.dataset={}
        self.complete_data=[]
        for speaker_name in self.speakers:
            speaker_root = os.path.join(self.data_root, speaker_name, 'clips')

            videos=[v for v in os.listdir(speaker_root) if not (v.endswith('mp4') or v.endswith('csv'))]
            print(videos)
        
            for vid in videos:
                source_vid=vid 
                vid_pth=os.path.join(speaker_root, source_vid, 'images/half', self.split)
                seqs = [s for s in os.listdir(vid_pth) if (s.startswith('clip'))]

                for s in seqs:
                    seq_root=os.path.join(vid_pth, s)
                    key = seq_root
                    audio_fname = os.path.join(speaker_root, source_vid, 'wavfiles/%s.wav'%(s))
                    if not os.path.isfile(audio_fname):
                        raise FileNotFoundError(audio_fname)

                    self.dataset[key]=PoseDataset(
                        data_root=seq_root,
                        speaker=speaker_name,
                        audio_fn=audio_fname,
                        
                        audio_sr=16000,
                        fps=25,
                        feat_method=feat_method,
                        audio_feat_dim=aud_feat_dim,
                        train=(self.split=='train'),
                        load_all=True,
                        split_trans_zero=self.split_trans_zero,
                        limbscaling=self.limbscaling,
                        num_frames=self.num_frames,
                        num_pre_frames=self.num_pre_frames,
                        audio_feat_win_size=aud_feat_win_size,
                        context_info=context_info
                    )
                    self.complete_data.append(self.dataset[key].complete_data)

        self.complete_data=np.concatenate(self.complete_data, axis=0)
        assert self.complete_data.shape[-1] == (12+21+21)*2
        self.normalize_stats = {}
        if self.normalization:
            data_mean, data_std = self._normalization_stats(self.complete_data)
            self.data_mean = data_mean.reshape(1, 1, -1)
            self.data_std = data_std.reshape(1, 1, -1)
        else:
            self.data_mean=None
            self.data_std = None

    def split_res(self):
        '''
        check labeling results for trans and zero 
        '''
        num_trans_frames = 0
        num_zero_frames = 0

        for key in list(self.dataset.keys()):
            num_trans_frames += self.dataset[key].trans_frames.shape[0]
            num_zero_frames += self.dataset[key].zero_frames.shape[0]
        
        print(self.complete_data.shape)
        print(num_trans_frames, num_zero_frames)
    
    def get_dataset(self):
        self.normalize_stats['mean'] = self.data_mean
        self.normalize_stats['std'] = self.data_std

        for key in list(self.dataset.keys()):
            self.dataset[key].get_dataset(self.normalization, self.normalize_stats)
            if self.split_trans_zero:
                if self.dataset[key].trans_dataset is not None:
                    self.trans_dataset_list.append(self.dataset[key].trans_dataset)
                if self.dataset[key].zero_dataset is not None:
                    self.zero_dataset_list.append(self.dataset[key].zero_dataset)
            else:
                self.all_dataset_list.append(self.dataset[key].all_dataset)
        
        if self.split_trans_zero:
            self.trans_dataset = data.ConcatDataset(self.trans_dataset_list)
            self.zero_dataset = data.ConcatDataset(self.zero_dataset_list)
        else:
            self.all_dataset = data.ConcatDataset(self.all_dataset_list)
    
    def _normalization_stats(self, complete_data):
        if self.norm_method == 'old':
            print('warning: using old data normalization')
            data_mean = np.mean(complete_data, axis=0)
            data_std = np.std(complete_data, axis=0)
            data_std[np.where(data_std==0)] = 1e-9
        elif self.norm_method == 'new':
            print('warning: using new data normalization')
            complete_data = complete_data.reshape(complete_data.shape[0], 54, 2).reshape(-1, 2)
            data_mean = np.mean(complete_data, axis=0)#(2)，should be(108)
            data_std = np.std(complete_data, axis=0)
            data_mean = data_mean.squeeze().reshape(1, 2)
            data_mean = np.repeat(data_mean, 54, 0)
            assert data_mean.shape[0] == 54 and data_mean.shape[1] == 2
            data_mean = data_mean.reshape(-1)

            data_std = data_std.squeeze().reshape(1, 2)
            data_std = np.repeat(data_std, 54, 0)
            assert data_std.shape[0] == 54 and data_std.shape[1] == 2
            data_std = data_std.reshape(-1)
            
            data_std[np.where(data_std==0)] = 1e-9 

        return data_mean, data_std

if __name__ == '__main__':
    d = MultiVidData(
        data_root='pose_dataset/videos/', 
        speakers=['Bill_Gates'], 
        split='train', 
        limbscaling=False, 
        normalization=False,
        split_trans_zero=True,
        num_frames=25,
        num_pre_frames=25,
        aud_feat_win_size=100,
        aud_feat_dim=64,
        feat_method='mel_spec',
        context_info=False
    )

    # d.split_res()
