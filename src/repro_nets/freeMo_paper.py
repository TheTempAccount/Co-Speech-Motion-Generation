import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from nets.layers import *
from nets.base import TrainWrapperBaseClass
from losses import KeypointLoss, KLLoss, VelocityLoss, L2RegLoss, AudioLoss
from nets.utils import parse_audio, denormalize
from data_utils import get_mfcc, get_melspec
import numpy as np

from repro_nets.freeMo_paper_code import LatentEncoder, LatentDecoder, SeqEncoder, SeqDecoder, Audio2Pose

class Generator(nn.Module):
    def __init__(self, 
        pose_dim,
        embed_size,
        noise_dim,
        content_dim,
        T_layer_norm,
        num_steps,
        seq_enc_hidden_size, 
        seq_dec_hidden_size,
        latent_enc_num_layers,
        latent_dec_num_layers,
        interaction,
        device,
        recon_input,
        aud_decoding,
        training=True,
        augmentation=False
    ):
        super().__init__()
        self.training = training
        self.content_dim = content_dim
        self.noise_dim = noise_dim
        self.recon_input = recon_input
        self.aud_decoding = aud_decoding
        self.device = device
        self.seq_encoder=SeqEncoder(
            pose_dim=pose_dim,
            embed_size=embed_size,
            content_dim=content_dim,
            T_layer_norm=T_layer_norm,
            hidden_size=seq_enc_hidden_size,
            device=device
        )
        self.seq_decoder=SeqDecoder(
            pose_dim=pose_dim,
            embed_size=embed_size,
            num_steps=num_steps,
            hidden_size = seq_dec_hidden_size,
            device=device
        )
        self.latent_encoder=LatentEncoder(
            embed_size = embed_size,
            noise_size= noise_dim,
            content_dim=content_dim,
            num_latent_fc_layers=latent_enc_num_layers,
            device=device
        )
        self.latent_decoder=LatentDecoder(
            embed_size=embed_size,
            noise_size=noise_dim,
            dec_hidden_size=seq_dec_hidden_size,
            content_dim=content_dim,
            dec_interaction=interaction,
            T_layer_norm=T_layer_norm,
            num_dec_fc_layers=latent_dec_num_layers,
            latent_fc_layers=latent_enc_num_layers,
            device=device
        )
        if self.aud_decoding:
            self.aud2pose = Audio2Pose(
                pose_dim=pose_dim,
                embed_size=embed_size,
                augmentation=augmentation
            )

    def forward(self, aud, pre_poses, gt_poses=None, mode='trans'):
        
        B, _, T = aud.shape

        
        pre_poses = pre_poses.permute(2, 0, 1)
        if gt_poses is not None:
            gt_poses = gt_poses.permute(2, 0, 1)

        if self.training:
            assert gt_poses is not None
            his_feat = self.seq_encoder(pre_poses)
            his_content, his_style = his_feat[:, :self.content_dim], his_feat[:, self.content_dim:]
            fut_feat = self.seq_encoder(gt_poses)
            fut_content, fut_style = fut_feat[:, :self.content_dim], fut_feat[:, self.content_dim:]

            mu, var, z = self.latent_encoder(his_style, fut_style)

            if mode == 'trans':
                pass
            elif mode == 'zero':
                z = torch.zeros_like(z).to(aud.device)
            elif mode == 'recon':
                raise ValueError("attemp to use recon in the main forwarding, where mode should be trans or zero")
            else:
                raise ValueError

            hidden, new_style = self.latent_decoder(z, his_content, his_style, data_type=mode)

            frame_0 = new_style

            pred_poses = self.seq_decoder(hidden, frame_0)
            pred_poses = pred_poses.permute(1, 2, 0)
            
            if self.recon_input:
                hidden_his, new_style = self.latent_decoder(None, his_content, his_style, data_type='recon')
                frame_0_his = new_style
                recon_poses_his = self.seq_decoder(hidden_his, frame_0_his)

                hidden_fut, new_style = self.latent_decoder(None, fut_content, fut_style, data_type='recon')
                frame_0_fut = new_style
                recon_poses_fut = self.seq_decoder(hidden_fut, frame_0_fut)

                recon_poses = (recon_poses_his.permute(1, 2, 0), recon_poses_fut.permute(1, 2, 0))
            else:
                recon_poses = None

            
            if mode == 'zero' and self.aud_decoding:
                dynamics = self.aud2pose(aud)

                pred_poses = pred_poses + dynamics
            else:
                dynamics = None

            return pred_poses, recon_poses, mu, var, dynamics
        else:
            his_feat = self.seq_encoder(pre_poses)
            his_content, his_style = his_feat[:, :self.content_dim], his_feat[:, self.content_dim:]

            if mode == 'trans':
                z = torch.randn([B, self.noise_dim]).to(aud.device)
            elif mode == 'zero':
                z = torch.zeros([B, self.noise_dim]).to(aud.device)
            elif mode == 'recon':
                z = None
            
            hidden, new_style = self.latent_decoder(z, his_content, his_style)
            frame_0 = new_style
            pred_poses,_ = self.seq_decoder(hidden, frame_0)
            pred_poses = pred_poses.permute(1, 2, 0)

            if mode == 'zero' and self.aud_decoding:
                dynamics = self.aud2pose(aud)
                pred_poses = pred_poses + dynamics

            return pred_poses

class TrainWrapper(TrainWrapperBaseClass):
    '''
    一个wrapper使其接受data_utils给出的数据，前向运算得到loss
    '''
    def __init__(self, args):
        self.args = args
        self.device = torch.device(self.args.gpu)
        self.generator = Generator(
            pose_dim=self.args.pose_dim,
            embed_size=self.args.embed_dim,
            noise_dim=self.args.noise_dim,
            content_dim=self.args.content_dim,
            T_layer_norm=self.args.T_layer_norm,
            num_steps=self.args.generate_length,
            seq_enc_hidden_size=self.args.seq_enc_hidden_size,
            seq_dec_hidden_size=self.args.seq_dec_hidden_size,
            latent_enc_num_layers=self.args.latent_enc_num_layers,
            latent_dec_num_layers=self.args.latent_dec_num_layers,
            interaction=self.args.interaction,
            device=self.device,
            recon_input=self.args.recon_input,
            aud_decoding=self.args.aud_decoding,
            training=(not self.args.infer),
            augmentation=False
        ).to(self.device)

        self.discriminator = None

        self.rec_loss = KeypointLoss().to(self.device)
        self.reg_loss = KeypointLoss().to(self.device)
        self.kl_loss = KLLoss(kl_tolerance=self.args.kl_tolerance).to(self.device)
        self.vel_loss = VelocityLoss(velocity_length=self.args.velocity_length).to(self.device)
        self.l2reg_loss = L2RegLoss().to(self.device)
        self.r_loss = AudioLoss().to(self.device)
        
        self.global_step = 0

        super().__init__(args)

    def __call__(self, bat):
        assert (not self.args.infer), "infer mode"
        self.global_step += 1

        trans_bat, zero_bat = bat[0], bat[1]

        
        trans_aud, trans_gt_poses, trans_pre_poses = trans_bat['aud_feat'].to(self.device).to(torch.float32), trans_bat['poses'].to(self.device).to(torch.float32), trans_bat['pre_poses'].to(torch.float32).to(self.device)

        trans_gt_conf = trans_bat['conf'].to(self.device).to(torch.float32)
    
        trans_pred_poses, trans_recon_poses, trans_mu, trans_var, trans_dynamics = self.generator(
            aud = trans_aud,
            pre_poses = trans_pre_poses,
            gt_poses = trans_gt_poses,
            mode = 'trans'
        )
        trans_loss, trans_dict = self.get_loss(
            trans_pred_poses, 
            trans_recon_poses, 
            trans_mu, 
            trans_var, 
            trans_dynamics, 
            trans_pre_poses, 
            trans_gt_poses, 
            mode='trans',
            gt_conf=trans_gt_conf)

        zero_aud, zero_gt_poses, zero_pre_poses = zero_bat['aud_feat'].to(self.device).to(torch.float32), zero_bat['poses'].to(self.device).to(torch.float32), zero_bat['pre_poses'].to(self.device).to(torch.float32)

        zero_gt_conf = zero_bat['conf'].to(self.device).to(torch.float32)

        zero_pred_poses, zero_recon_poses, zero_mu, zero_var, zero_dynamics = self.generator(
            aud = zero_aud,
            pre_poses = zero_pre_poses,
            gt_poses = zero_gt_poses,
            mode = 'zero'
        )

        zero_loss, zero_dict = self.get_loss(
            zero_pred_poses, 
            zero_recon_poses, 
            zero_mu, 
            zero_var, 
            zero_dynamics, 
            zero_pre_poses, 
            zero_gt_poses, 
            mode='zero',
            gt_conf=zero_gt_conf
            )

        
        total_loss = trans_loss + zero_loss
        loss_dict = {}
        for key in list(trans_dict.keys()):
            loss_dict[key] = trans_dict[key] + zero_dict[key]

        self.generator_optimizer.zero_grad()
        total_loss.backward()
        grad = torch.nn.utils.clip_grad_norm(self.generator.parameters(), self.args.max_gradient_norm)
        loss_dict['grad'] = grad
        self.generator_optimizer.step()
        
        return total_loss, loss_dict

    def get_loss(self, 
            pred_poses, 
            recon_poses,
            mu, 
            var, 
            dynamics,
            prev_poses, 
            gt_poses,  
            mode='trans',
            gt_conf=None
        ):
        loss_dict = {}
        if mode == 'trans':
            recon_loss = self.rec_loss(pred_poses, gt_poses, gt_conf)
            if recon_poses is not None:
                
                reg_loss = self.reg_loss(recon_poses[0], prev_poses) + self.reg_loss(recon_poses[1], gt_poses)
            else:
                reg_loss = 0

            vae_loss = self.kl_loss(mu, var)
            vel_loss = self.vel_loss(pred_poses, gt_poses, prev_poses)
            r_loss = 0

            curr_velocity_weight=(self.args.vel_loss_weight - (self.args.vel_loss_weight - self.args.vel_start_weight)*(self.args.vel_decay_rate)**self.global_step)
            
            curr_kl_weight=(self.args.kl_loss_weight - (self.args.kl_loss_weight - self.args.kl_start_weight)*(self.args.kl_decay_rate)**self.global_step)

            total_loss = self.args.keypoint_loss_weight * recon_loss + \
                        self.args.recon_input_weight * reg_loss + \
                            curr_kl_weight * vae_loss + \
                                self.args.keypoint_loss_weight * curr_velocity_weight * vel_loss + \
                                    self.args.r_loss_weight * r_loss
            
            loss_dict['recon_loss'] = recon_loss
            loss_dict['reg_loss'] = reg_loss
            loss_dict['vae_loss'] = vae_loss
            loss_dict['vel_loss'] = vel_loss
            loss_dict['r_loss'] = r_loss
        elif mode == 'zero':
            recon_loss = self.rec_loss(pred_poses, gt_poses)
            if recon_poses is not None:
                reg_loss = self.reg_loss(recon_poses[0], prev_poses) + self.reg_loss(recon_poses[1], gt_poses)
            else:
                reg_loss = 0

            if self.args.zero_loss_weight > 0:
                vae_loss = self.l2reg_loss(mu) + self.l2reg_loss(var)
            else:
                vae_loss = 0

            vel_loss = self.vel_loss(pred_poses, gt_poses, prev_poses)
            curr_velocity_weight=(self.args.vel_loss_weight - (self.args.vel_loss_weight - self.args.vel_start_weight)*(self.args.vel_decay_rate)**self.global_step)

            if dynamics is not None:
                r_loss = self.r_loss(dynamics, gt_poses)
            else:
                r_loss = 0

            total_loss = self.args.keypoint_loss_weight * recon_loss + \
                        self.args.recon_input_weight * reg_loss + \
                            self.args.zero_loss_weight * vae_loss + \
                                self.args.keypoint_loss_weight * curr_velocity_weight * vel_loss + \
                                    self.args.r_loss_weight * r_loss
            
            loss_dict['recon_loss'] = recon_loss
            loss_dict['reg_loss'] = reg_loss
            loss_dict['vae_loss'] = vae_loss
            loss_dict['vel_loss'] = vel_loss
            loss_dict['r_loss'] = r_loss
        else:
            raise ValueError

        return total_loss, loss_dict
    
    def infer_on_audio(self, aud_fn, initial_pose=None, norm_stats=None, **kwargs):
        '''
        initial_pose: (B, C, T)
        对aud和txgfile这一栏目生成一个(B, T, C)的序列
        如果是normalization的模型，则要求initial_pose也是经过normalize的
        '''
        output = []

        assert self.args.infer, "train mode"
        self.generator.eval()

        if self.args.normalization:
            assert norm_stats is not None
            data_mean = norm_stats[0]
            data_std = norm_stats[1]

        pre_length = self.args.pre_pose_length
        generate_length = self.args.generate_length
        assert initial_pose.shape[-1] == pre_length

        txgfile = kwargs['txgfile']
        code_seq = parse_audio(txgfile)
        if self.args.feat_method == 'mel_spec':
            aud_feat = get_melspec(aud_fn).transpose(1, 0)
        elif self.args.feat_method == 'mfcc':
            aud_feat = get_mfcc(
                audio_fn=aud_fn,
                n_mfcc=self.args.aud_feat_dim,
                win_size=self.args.aud_feat_win_size
            )
        else:
            raise ValueError(self.args.feat_method)


        for step in range(len(code_seq)):
            if self.args.aud_feat_win_size is None:
                aud_feat_step = aud_feat[:, step*generate_length: (step+1)*generate_length]
                if aud_feat_step.shape[-1] < generate_length:
                    aud_feat_step = np.pad(aud_feat_step, [[0, 0], [0, generate_length-aud_feat_step.shape[-1]]], mode='constant')
            else:
                aud_feat_step = aud_feat[:, step*self.args.aud_feat_win_size:(step+1)*self.args.aud_feat_win_size]
                if aud_feat_step.shape[-1] < self.args.aud_feat_win_size:
                    aud_feat_step = np.pad(aud_feat_step, [[0, 0], [0, self.args.aud_feat_win_size-aud_feat_step.shape[-1]]], mode='constant')

            aud_feat_step = aud_feat_step[np.newaxis, :].repeat(initial_pose.shape[0], axis=0)
            aud_feat_step = torch.tensor(aud_feat_step, dtype = torch.float32).to(self.device)

            with torch.no_grad():
                code = code_seq[step]
                if code == 1:
                    pred_poses = self.generator(aud_feat_step, initial_pose, mode='trans') 
                elif code==0:
                    pred_poses = self.generator(aud_feat_step, initial_pose, mode='zero')
                else: raise ValueError

            initial_pose = pred_poses[:, :, -pre_length:]

            output_step = pred_poses.clone().cpu().detach().numpy().transpose(0, 2, 1) 
            if self.args.normalization:
                output_step = denormalize(output_step, data_mean, data_std)
            output.append(output_step)

        output = np.concatenate(output, axis=1)
        print(output.shape)
        print(code_seq)
        return output    