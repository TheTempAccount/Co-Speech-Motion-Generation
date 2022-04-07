import os
import sys

sys.path.append(os.getcwd())

from data_utils import torch_data

from trainer.options import parse_args
from trainer.config import load_JsonConfig
from nets import *
from repro_nets import *

import torch
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import random
import logging
import time
import shutil


class Trainer():
    def __init__(self) -> None:
        parser = parse_args()
        self.args = parser.parse_args()

        self.device = torch.device(self.args.gpu)
        torch.cuda.set_device(self.device)
        self.setup_seed(self.args.seed)
        self.set_train_dir()

        shutil.copy(self.args.config_file, self.train_dir)
        
        self.config = load_JsonConfig(self.args.config_file)
        
        self.init_dataloader()
        self.init_model()
        self.start_epoch = 0
        self.global_steps = 0
        if self.args.resume:
            self.resume()
        # self.init_optimizer()

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def set_train_dir(self):
        time_stamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        train_dir= os.path.join(self.args.save_dir, os.path.normpath(time_stamp+'-'+self.args.exp_name+'-'+time.strftime("%H:%M:%S")))
        os.makedirs(train_dir, exist_ok=True)
        log_file=os.path.join(train_dir, 'train.log')

        fmt="%(asctime)s-%(lineno)d-%(message)s"
        logging.basicConfig(
            stream=sys.stdout, level=logging.INFO,format=fmt, datefmt='%m/%d %I:%M:%S %p'
        )
        fh=logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)
        self.train_dir = train_dir

    def resume(self):
        print('resume from a previous ckpt')
        ckpt = torch.load(self.args.pretrained_pth)
        self.generator.load_state_dict(ckpt['generator'])
        self.start_epoch = ckpt['epoch']
        self.global_steps = ckpt['global_steps']
        self.generator.global_step = self.global_steps

    def init_model(self):
        if self.config.Model.model_name == 'freeMo':
            # if not self.args.context_info:
            #     self.generator = freeMo_Generator(
            #         self.args
            #     )
            # else:
            #     self.generator = new_aud_Generator(
            #         self.args
            #     )
            self.generator = freeMo_dev(
                self.args,
                self.config
            )
        elif self.config.Model.model_name == 'S2G':
            self.generator = S2G_Generator(
                    self.args,
                    self.config,
                )
        elif self.config.Model.model_name == 'Tmpt':
            self.generator = S2G_Generator(
                    self.args,
                    self.config,
                )
        elif self.config.Model.model_name == 'tricon':
            raise NotImplementedError
        elif self.config.Model.model_name == 'a2b':
            raise NotImplementedError
        elif self.config.Model.model_name == 'mix_stage':
            raise NotImplementedError
        elif self.config.Model.model_name == 'StyleGestures':
            self.generator = StyleGesture_Generator(
                self.args,
                self.config
            )
        elif self.config.Model.model_name == 'Audio2Gestures':
            self.generator = Audio2Gesture_Generator(
                self.args,
                self.config,
                self.train_set.data_mean,
                self.train_set.data_std
            )
        else:
            raise ValueError

    def init_dataloader(self):
        if 'freeMo' in self.config.Model.model_name:
            if self.config.Data.data_root.endswith('.csv'):
                raise NotImplementedError
            else:
                data_class = torch_data
            
            self.train_set = data_class(
                data_root=self.config.Data.data_root,
                speakers=self.args.speakers,
                split='train',
                limbscaling=self.config.Data.pose.augmentation,
                normalization=self.config.Data.pose.normalization,
                norm_method=self.config.Data.pose.norm_method,
                split_trans_zero=True,
                num_pre_frames=self.config.Data.pose.pre_pose_length,
                num_frames=self.config.Data.pose.generate_length,
                aud_feat_win_size=self.config.Data.aud.aud_feat_win_size,
                aud_feat_dim=self.config.Data.aud.aud_feat_dim,
                feat_method=self.config.Data.aud.feat_method,
                context_info=self.config.Data.aud.context_info
            )

            if self.config.Data.pose.normalization:
                self.norm_stats = (self.train_set.data_mean, self.train_set.data_std)
                save_file = os.path.join(self.train_dir, 'norm_stats.npy')
                np.save(save_file, self.norm_stats, allow_pickle=True)

            self.train_set.get_dataset()
            self.trans_set = self.train_set.trans_dataset
            self.zero_set = self.train_set.zero_dataset

            self.trans_loader = data.DataLoader(self.trans_set, batch_size=self.config.DataLoader.batch_size, shuffle=True, num_workers=self.config.DataLoader.num_workers, drop_last=True) 
            self.zero_loader = data.DataLoader(self.zero_set, batch_size=self.config.DataLoader.batch_size, shuffle=True, num_workers=self.config.DataLoader.num_workers, drop_last=True)
            
        else:
            data_class = torch_data
            
            self.train_set = data_class(
                data_root=self.config.Data.data_root,
                speakers=self.args.speakers,
                split='train',
                limbscaling=self.config.Data.pose.augmentation,
                normalization=self.config.Data.pose.normalization,
                norm_method=self.config.Data.pose.norm_method,
                split_trans_zero=False,
                num_pre_frames=self.config.Data.pose.pre_pose_length,
                num_frames=self.config.Data.pose.generate_length,
                aud_feat_win_size=self.config.Data.aud.aud_feat_win_size,
                aud_feat_dim=self.config.Data.aud.aud_feat_dim,
                feat_method=self.config.Data.aud.feat_method,
                context_info=self.config.Data.aud.context_info
            )

            if self.config.Data.pose.normalization:
                self.norm_stats = (self.train_set.data_mean, self.train_set.data_std)
                save_file = os.path.join(self.train_dir, 'norm_stats.npy')
                np.save(save_file, self.norm_stats, allow_pickle=True)

            self.train_set.get_dataset()

            self.train_loader = data.DataLoader(self.train_set.all_dataset, batch_size=self.config.DataLoader.batch_size, shuffle=True, num_workers=self.config.DataLoader.num_workers, drop_last=True) 
            

    def init_optimizer(self):
        pass

    def print_func(self, loss_dict, steps):
        info_str = ['global_steps:%d'%(self.global_steps)]
        info_str += ['%s:%.4f'%(key, loss_dict[key]/steps) for key in list(loss_dict.keys())]
        logging.info(','.join(info_str))
    
    def save_model(self, epoch):
        state_dict = {
            'generator': self.generator.state_dict(),
            'epoch': epoch,
            'global_steps': self.global_steps
        }
        save_name = os.path.join(self.train_dir, 'ckpt-%d.pth'%(epoch))
        torch.save(state_dict, save_name)

    def train_epoch(self):
        epoch_loss_dict = {} #最好是追踪每个epoch的loss变换
        epoch_steps = 0
        if 'freeMo' in self.config.Model.model_name:
            for bat in zip(self.trans_loader, self.zero_loader):
                self.global_steps += 1
                epoch_steps += 1
                _, loss_dict = self.generator(bat)
                
                if epoch_loss_dict:#非空
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] += loss_dict[key]
                else:
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] = loss_dict[key]

                if self.global_steps % self.config.Log.print_every == 0:
                    self.print_func(epoch_loss_dict, epoch_steps)
        else:
            for bat in self.train_loader:
                self.global_steps += 1
                epoch_steps += 1
                _, loss_dict = self.generator(bat)
                if epoch_loss_dict:#非空
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] += loss_dict[key]
                else:
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] = loss_dict[key]
                if self.global_steps % self.config.Log.print_every == 0:
                    self.print_func(epoch_loss_dict, epoch_steps)

    def train(self):
        logging.info('start_training')
        self.total_loss_dict = {}
        for epoch in range(self.start_epoch, self.config.Train.epochs):
            logging.info('epoch:%d'%(epoch))
            self.train_epoch()
            if (epoch+1)%self.config.Log.save_every == 0 and epoch > 30:
                self.save_model(epoch)
    
