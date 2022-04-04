import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from nets.base import TrainWrapperBaseClass
from nets.glow.utils import save
from nets.glow.config import JsonConfig
from nets.glow.models import Glow
from nets.glow import thops
from nets.glow.builder import build
from nets.utils import denormalize
from data_utils import get_mfcc_psf

class Generator(nn.Module):
    def __init__(self,
        pose_dim,
        num_pre_poses,#
        n_lookahead,#
        graph,#
        global_step,
        training = True 
    ) -> None:
        super().__init__()
        self.pose_dim = pose_dim
        self.num_pre_poses = num_pre_poses
        self.n_lookahead = n_lookahead

        #模型的初始化
        self.graph = graph
        self.global_step = global_step

        self.training = training
    
    def forward(self, aud, pre_poses, gt_poses=None):
        '''
        前向运算得到结果
        '''
        if self.training:
            #aud:(B, aud_dim, T_1)
            #pre_poses: (B, pose_dim, T_0)
            #gt_poses: (B, pose_dim, T_0)

            #constructing x: (B, C_1, T) and cond: (B, C_2, T)。It may takes 0.05s
            poses = torch.cat([pre_poses, gt_poses], dim=2)
            #TODO: hard codeed
            assert poses.shape[1] == self.pose_dim and poses.shape[2] == 50 and aud.shape[2] == 75 
            
            if poses.shape[2] < aud.shape[2]:
                # poses = F.pad(poses, (0, aud.shape[2]-poses.shape[2]), mode='constant')
                aud = aud[..., :poses.shape[2]+self.n_lookahead]
            
            num_steps = np.min([poses.shape[2]-self.num_pre_poses, aud.shape[2]-self.n_lookahead-self.num_pre_poses])
            assert num_steps == 45

            #pre_poses
            frame_start = 0
            frame_end = aud.shape[2] - self.n_lookahead -self.num_pre_poses -1
            assert frame_end-frame_start+1 == num_steps

            pre_pose_idx = np.zeros([num_steps, self.num_pre_poses])
            for i in range(self.num_pre_poses):
                pre_pose_idx[:, i] = (np.arange(frame_start, frame_end+1)+i).transpose()
            
            pre_pose_cond = poses[:, :, pre_pose_idx].permute(0, 1, 3, 2)#(B, C, num_pre_pose, num_steps)
            pre_pose_cond = pre_pose_cond.contiguous().view(pre_pose_cond.shape[0], -1, pre_pose_cond.shape[-1])#(B, C, T)

            #aud
            aud_idx = np.zeros([num_steps, self.num_pre_poses+1+self.n_lookahead])
            for i in range(self.num_pre_poses+1+self.n_lookahead):
                aud_idx[:, i] = (np.arange(frame_start, frame_end+1)+i).transpose()
            aud_cond = aud[:, :, aud_idx].permute(0, 1, 3, 2)
            aud_cond = aud_cond.contiguous().view(aud_cond.shape[0], -1, aud_cond.shape[-1]) #(B, C, T)

            cond = torch.cat([pre_pose_cond, aud_cond], dim=1) #(B, C, T)

            #x
            x_idx = np.zeros([num_steps, 1])
            x_idx[:, 0] = (np.arange(frame_start, frame_end+1)+self.num_pre_poses).transpose()

            x = poses[:, :, x_idx].permute(0, 1, 3, 2)
            x = x.contiguous().view(x.shape[0], -1, x.shape[-1]) #(B, C, T)

            #with x and cond, do forwarding

            self.graph.init_lstm_hidden() 

            #initialization actNorm
            if self.global_step == 0:
                self.graph(x, cond)
                self.graph.init_lstm_hidden()

            # forward
            z, nll = self.graph(x=x, cond=cond)

            self.global_step += 1

            return z, nll
        
        else:
            assert pre_poses.shape[-1] == self.num_pre_poses and aud.shape[-1] == self.num_pre_poses + 1 + self.n_lookahead
            pre_poses = pre_poses.contiguous().view(pre_poses.shape[0], -1, 1) #注意这里的reshape时候的维度顺序，C在前面，和training的时候保持一致
            aud = aud.view(aud.shape[0], -1, 1)
            cond = torch.cat([pre_poses, aud], dim=1) #(B, C, 1)

            sampled = self.graph(z=None, cond=cond, eps_std=1.0, reverse=True)
            # sampled = sampled.cpu().numpy()[:,:,0]#(B, C) generated pose

            return sampled

class TrainWrapper(TrainWrapperBaseClass):
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step=0
        hparams = args.config_file
        hparams = JsonConfig(hparams)
        self.hparams = hparams
        # is_training = hparams.Infer.pre_trained == ""
        is_training = not self.args.infer
        # if is_training:
        #     assert not self.args.infer

        x_channels = self.config.Data.pose.pose_dim
        cond_channels = (hparams.Data.seqlen + 1 + hparams.Data.n_lookahead) * 64 + hparams.Data.seqlen * 108

        built = build(
            x_channels=x_channels,
            cond_channels=cond_channels,
            hparams=hparams,
            is_training=is_training
        )
        self.built = built
        self.generator = Generator(
            pose_dim=self.config.Data.pose.pose_dim,
            num_pre_poses=hparams.Data.seqlen,
            n_lookahead=hparams.Data.n_lookahead,
            graph = self.built['graph'],
            global_step=self.global_step,
            training=is_training
        ).to(self.device)

        self.discriminator = None

        self.lrschedule = self.built['lrschedule']
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm

        super().__init__(args, config)

        # self.init_optimizer()

    def init_optimizer(self) -> None:
        '''
        rewrite
        '''
        self.generator_optimizer = self.built['optim']
    
    def __call__(self, bat):
        '''
        batch -> loss
        '''
        assert (not self.args.infer), "infer mode"
        self.generator.train()
        lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
        self.global_step += 1

        #更新lr
        for param_group in self.generator_optimizer.param_groups:
            param_group['lr'] = lr
        
        self.generator_optimizer.zero_grad()

        #获取数据
        aud, gt_poses, pre_poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32), bat['pre_poses'].to(torch.float32).to(self.device)

        z, nll = self.generator(aud, gt_poses, pre_poses)

        total_loss, loss_dict = self.get_loss(nll)
        
        #backward and update
        self.generator.zero_grad()
        self.generator_optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_clip is not None and self.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(self.generator.parameters(), self.max_grad_clip)

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)

        self.generator_optimizer.step()

        loss_dict['grad'] = grad_norm
        return total_loss, loss_dict

    def get_loss(self, nll):
        loss_dict={}
        loss_generative = Glow.loss_generative(nll)
        loss_classes = 0
        loss = loss_generative
        loss_dict['loss_generative'] = loss
        return loss, loss_dict

    def infer_on_audio(self, aud_fn, initial_pose=None, norm_stats=None, **kwargs):
        '''
        we initialise initial_pose from outside and specify batch dimension
        '''
        output = []
        assert self.args.infer, "train mode"
        self.generator.eval()
        if self.config.Data.pose.normalization:
            assert norm_stats is not None
            data_mean = norm_stats[0]
            data_std = norm_stats[1]

        aud_feat = get_mfcc_psf(aud_fn).transpose(1, 0)#(C, T)
        initial_pose = initial_pose[:, :, -self.hparams.Data.seqlen:]

        assert initial_pose.shape[-1] == self.hparams.Data.seqlen and initial_pose.shape[1] == 108

        total_steps = aud_feat.shape[-1] - self.hparams.Data.n_lookahead - self.hparams.Data.seqlen
        for step in range(total_steps):
            '''
            frame-by-frame
            '''
            aud_feat_step = aud_feat[:, step:step + self.hparams.Data.n_lookahead + self.hparams.Data.seqlen + 1]
            aud_feat_step = aud_feat_step[np.newaxis, :].repeat(initial_pose.shape[0], axis=0)
            aud_feat_step = torch.tensor(aud_feat_step, dtype = torch.float32).to(self.device) #(B, C, T)

            sample = self.generator(aud=aud_feat_step, pre_poses = initial_pose) #(B, C)

            initial_pose = torch.cat([initial_pose[:, :, 1:], sample], dim=-1)

            output_step = sample.clone().cpu().detach().numpy().transpose(0, 2, 1) #(B, 1, C)
            if self.config.Data.pose.normalization:
                output_step = denormalize(output_step, data_mean, data_std)
            output.append(output_step)

        output = np.concatenate(output, axis=1) #(B, total_steps, C)
        print(output.shape)
        return output  

if __name__ == "__main__":
    '''
    test data transformation
    '''
    # import time
    # aud = torch.randn([64, 64, 75])
    # pre_poses = torch.randn([64, 108, 25])
    # gt_poses = torch.randn([64, 108, 25])

    # start = time.time()
    # n_lookahead = 20
    # num_pre_poses = 5

    # poses = torch.cat([pre_poses, gt_poses], dim=2)
    #     #TODO: hard codeed
    # assert poses.shape[1] == 108 and poses.shape[2] == 50 and aud.shape[2] == 75 
        
    # if poses.shape[2] < aud.shape[2]:
    #     # poses = F.pad(poses, (0, aud.shape[2]-poses.shape[2]), mode='constant')
    #     aud = aud[..., :poses.shape[2]+n_lookahead]
        
    # #总共能够多少个step
    # num_steps = np.minimum(poses.shape[2]-num_pre_poses, aud.shape[2]-n_lookahead-num_pre_poses)
    # assert num_steps == 45

    # #这个是pre_poses的起始位置
    # frame_start = 0
    # frame_end = aud.shape[2] - n_lookahead - 1 -num_pre_poses
    # assert frame_end-frame_start+1 == num_steps

    # pre_pose_idx = np.zeros([num_steps, num_pre_poses])
    # for i in range(num_pre_poses):
    #     pre_pose_idx[:, i] = (np.arange(frame_start, frame_end+1)+i).transpose()
        
    # pre_pose_cond = poses[:, :, pre_pose_idx].permute(0, 1, 3, 2)#(B, C, num_pre_pose, num_steps)
    # pre_pose_cond = pre_pose_cond.contiguous().view(pre_pose_cond.shape[0], -1, pre_pose_cond.shape[-1])#(B, C, T)

    #     #aud特征的起始位置
    # aud_idx = np.zeros([num_steps, num_pre_poses+1+n_lookahead])
    # for i in range(num_pre_poses+1+n_lookahead):
    #     aud_idx[:, i] = (np.arange(frame_start, frame_end+1)+i).transpose()
    # aud_cond = aud[:, :, aud_idx].permute(0, 1, 3, 2)
    # aud_cond = aud_cond.contiguous().view(aud_cond.shape[0], -1, aud_cond.shape[-1]) #(B, C, T)

    # cond = torch.cat([pre_pose_cond, aud_cond], dim=1) #(B, C, T)

    # #构建x
    # x_idx = np.zeros([num_steps, 1])
    # x_idx[:, 0] = (np.arange(frame_start, frame_end+1)+num_pre_poses).transpose()

    # x = poses[:, :, x_idx].permute(0, 1, 3, 2)
    # x = x.contiguous().view(x.shape[0], -1, x.shape[-1]) #(B, C, T)

    # end = time.time()

    # print(cond.shape, x.shape, end-start)

    # batch_random_idx = np.random.randint(0, cond.shape[0])
    # random_step = np.random.randint(0, cond.shape[-1])
    # cond_sample = cond[batch_random_idx, :, random_step]
    # x_sample = x[batch_random_idx, :, random_step]#(108)

    # assert (x_sample.squeeze() == poses[batch_random_idx, :, random_step+num_pre_poses].squeeze()).numpy().all()
    # gt_pose_cond = poses[batch_random_idx, :, random_step:random_step+num_pre_poses].reshape(-1).squeeze()
    # gt_aud_cond = aud[batch_random_idx, :, random_step:random_step+num_pre_poses+1+n_lookahead].reshape(-1).squeeze()
    # gt_cond = torch.cat([gt_pose_cond, gt_aud_cond]).squeeze()
    # assert (cond_sample == gt_cond).squeeze().numpy().all()

    # hparams = JsonConfig('./config/style_gestures.json')

    # cond_channels = (5 + 1 + 25) * 64 + 5 * 108

    # built = build(
    #     x_channels=108,
    #     cond_channels=cond_channels,
    #     hparams=hparams,
    #     is_training=True
    # )

    # test_model = Generator(
    #     pose_dim=108,
    #     num_pre_poses=5,
    #     n_lookahead=25,
    #     graph=built['graph'],
    #     global_step=0,
    #     training=True
    # ).cuda()

    # aud = torch.random.randn([64, 64, 75]).cuda()
    # pre_poses = torch.random.randn([64, 108, 25]).cuda()
    # gt_poses = torch.random.randn([64, 108, 25]).cuda()

    # z, nll = test_model(aud, pre_poses, gt_poses)

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--pose_dim', default=108, type=int)
    parser.add_argument('--aud_dim', default=64, type=int)
    parser.add_argument('--config_file', default='./config/style_gestures.json', type=str)
    args = parser.parse_args()

    trainer = TrainWrapper(args)

    aud = torch.randn([64, 64, 75]).cuda()
    pre_poses = torch.randn([64, 108, 25]).cuda()
    gt_poses = torch.randn([64, 108, 25]).cuda()

    bat={
        'aud_feat': aud,
        'poses': gt_poses,
        'pre_poses': pre_poses
    }

    total_loss, loss_dict = trainer(bat)

    print(total_loss)
    print()
    print(loss_dict)