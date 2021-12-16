import os
import sys

from numpy.core.defchararray import split
sys.path.append(os.getcwd())

import torch
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import random
import logging
import time
import shutil

from trainer.options import parse_args
from nets import freeMo_Generator, S2G_Generator, A2B_Generator, TriCon_Generator
from data_utils import csv_data, torch_data

class Trainer():
    def __init__(self) -> None:
        parser = parse_args()
        self.args = parser.parse_args()

        self.device = torch.device(self.args.gpu)
        torch.cuda.set_device(self.device)
        self.setup_seed(self.args.seed)
        self.set_train_dir()

        
        assert self.args.shell_cmd is not None, "save the shell cmd"
        shutil.copy(self.args.shell_cmd, self.train_dir)

        self.init_model()
        self.init_dataloader()
        self.init_optimizer()

        self.global_steps = 0

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

    def init_model(self):
        if self.args.model_name == 'freeMo':
            self.generator = freeMo_Generator(
                self.args
            )
            self.discriminator = None

        elif self.args.model_name == 's2g':
            raise NotImplementedError
        elif self.args.model_name == 'tmpt':
            raise NotImplementedError
        elif self.args.model_name == 'tricon':
            raise NotImplementedError
        elif self.args.model_name == 'a2b':
            raise NotImplementedError
        elif self.args.model_name == 'mix_stage':
            raise NotImplementedError
        else:
            raise ValueError

    def init_dataloader(self):
        if self.args.model_name == 'freeMo':
            if self.args.data_root.endswith('.csv'):
                raise NotImplementedError
            else:
                data_class = torch_data
            
            self.train_set = data_class(
                data_root=self.args.data_root,
                speakers=self.args.speakers,
                split='train',
                limbscaling=self.args.augmentation,
                normalization=self.args.normalization,
                split_trans_zero=True,
                num_pre_frames=self.args.pre_pose_length,
                num_frames=self.args.generate_length,
                aud_feat_win_size=self.args.aud_feat_win_size,
                aud_feat_dim=self.args.aud_feat_dim,
                feat_method=self.args.feat_method
            )

            if self.args.normalization:
                self.norm_stats = (self.train_set.data_mean, self.train_set.data_std)
                save_file = os.path.join(self.train_dir, 'norm_stats.npy')
                np.save(save_file, self.norm_stats, allow_pickle=True)

            self.train_set.get_dataset()
            self.trans_set = self.train_set.trans_dataset
            self.zero_set = self.train_set.zero_dataset

            self.trans_loader = data.DataLoader(self.trans_set, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True) 
            self.zero_loader = data.DataLoader(self.zero_set, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
            
            
            

        else:
            raise NotImplementedError

    def init_optimizer(self):
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr = self.args.generator_learning_rate,
            betas=[0.9, 0.999]
        )
        if self.discriminator is not None:
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr = self.args.discriminator_learning_rate,
                betas=[0.9, 0.999]
            )

    def print_func(self, loss_dict):
        info_str = ['global_steps:%d'%(self.global_steps)]
        info_str += ['%s:%.4f'%(key, loss_dict[key]) for key in list(loss_dict.keys())]
        logging.info(','.join(info_str))
    
    def save_model(self, epoch):
        state_dict = {
            'generator': self.generator.state_dict(),
            'epoch': epoch,
            'global_steps': self.global_steps,
            'generator_optim': self.generator_optimizer.state_dict(),
            'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
            'discriminator_optim': self.discriminator_optimizer.state_dict() if self.discriminator is not None else None
        }
        save_name = os.path.join(self.train_dir, 'ckpt-%d.pth'%(epoch))
        torch.save(state_dict, save_name)

    def train_epoch(self):
        
        
        for bat in zip(self.trans_loader, self.zero_loader):
            self.global_steps += 1
            _, loss_dict = self.generator(bat)

            if self.global_steps % self.args.print_every == 0:
                self.print_func(loss_dict)

    def train(self):
        logging.info('start_training')
        for epoch in range(self.args.epochs):
            logging.info('epoch:%d'%(epoch))
            self.train_epoch()
            if (epoch+1)%self.args.save_every == 0 and epoch > 30:
                self.save_model(epoch)
    
