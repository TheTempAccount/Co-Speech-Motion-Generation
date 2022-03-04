import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from nets.layers import *
from nets.graph_definition import *
from losses import KeypointLoss, KLLoss, VelocityLoss, L2RegLoss, AudioLoss
from nets.utils import parse_audio, denormalize
from nets.base import TrainWrapperBaseClass
from data_utils import get_mfcc, get_melspec
import numpy as np


class SeqEncoderWrapper(nn.Module):
    '''
    使用graph进行seq的encoder
    '''
    def __init__(self,
        A,
        T,
        embed_size,
        layer_configs,
        residual=True,
        local_bn=False,
        share_weights=False
    ):
        super(SeqEncoderWrapper, self).__init__()
        self.seq_enc = SeqEncoderGraph(
            embedding_size=embed_size,
            layer_configs=layer_configs,
            residual=residual,
            local_bn=local_bn,
            A = A,
            T = T,
            share_weights=share_weights
        )
    
    def forward(self, x):
        x = self.seq_enc(x)
        return x

class SeqDecoderWrapper(nn.Module):
    '''
    LSTM-FC. 
    '''
    def __init__(self,
        hidden_size,
        in_size,
        out_size,
        num_steps,
        num_layers,
        rnn_cell='lstm'
    ):
        super(SeqDecoderWrapper, self).__init__()

        self.num_steps = num_steps
        if rnn_cell == 'lstm':
            self.rnn_cell = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_cell == 'gru':
            self.rnn_cell = nn.GRU(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
    
    def forward(self, x, frame_0):
        frame_0 = frame_0.squeeze()
        if len(frame_0.shape)==2:
            frame_0 = frame_0.unsqueeze(1)

        outputs = []
        prev_states = x
        dec_input = frame_0
        for _ in range(self.num_steps):
            output, prev_states = self.rnn_cell(dec_input, prev_states)
            outputs.append(output.squeeze(1))
        outputs = torch.stack(outputs, dim=1)
        outputs = self.fc(outputs)
        return outputs.permute(0, 2, 1)

class LatentEncoderWrapper(nn.Module):
    def __init__(self,
        embed_size,
        content_size,
        fc_size,
        noise_size,
        num_latent_fc_layers,
        interaction_method='add'
    ):
        super(LatentEncoderWrapper, self).__init__()
        self.content_size = content_size
        embed_size = embed_size - content_size
        self.interaction_method = interaction_method
        if self.interaction_method == 'concat':
            self.pre_fc = nn.Sequential(
                nn.Linear(embed_size*2, embed_size),
                nn.ReLU()
            )

        self.layer_norm_1 = nn.LayerNorm(embed_size)

        latent_fc_layers = []
        latent_fc_layers.append(
            nn.Sequential(ResBlock(
                    input_dim=embed_size,
                    fc_dim=fc_size,
                    afn='relu',
                    nfn=None
                ),
                nn.LayerNorm(fc_size)
            )
        )

        for i in range(num_latent_fc_layers-1):
            latent_fc_layers.append(
                nn.Sequential(ResBlock(
                    input_dim=fc_size,
                    fc_dim=fc_size,
                    afn='relu',
                    nfn=None
                ),
                nn.LayerNorm(fc_size)
                )
            )
        self.latent_fc_layers = nn.Sequential(*latent_fc_layers)

        self.mu_fc = nn.Linear(fc_size, noise_size)
        self.var_fc = nn.Linear(fc_size, noise_size)

    def forward(self, his_feat, fut_feat):
        
        
        his_feat = his_feat[:, self.content_size:]
        fut_feat = fut_feat[:, self.content_size:]

        if self.interaction_method == 'concat':
            
            feat_dis = torch.cat([fut_feat, his_feat], dim=1)
            feat_dis = self.pre_fc(feat_dis)
        elif self.interaction_method == 'add':
            
            feat_dis = fut_feat - his_feat
        else:
            raise ValueError

        
        x = self.layer_norm_1(feat_dis)
        
        x = self.latent_fc_layers(x)
        mu = self.mu_fc(x)
        var = self.var_fc(x)
        sampled_z = self.sample(mu, var)
        return mu, var, sampled_z
    
    def sample(self, mu, var):
        std=torch.exp(0.5*var)
        eps=torch.randn_like(std, device=mu.device)
        z=eps*std+mu
        return z

class LatentDecoderWrapper(nn.Module):
    def __init__(self,
        embed_size,
        content_size,
        noise_size,
        num_latent_fc_layers,
        num_dec_fc_layers,
        dec_interaction,
        dec_hidden_size,
        T_layer_norm=False,
        dec_type='gru'
    ):
        super(LatentDecoderWrapper, self).__init__()
        self.content_size = content_size
        embed_size -= content_size

        self.embed_size = embed_size
        self.noise_size = noise_size
        self.num_latent_fc_layers = num_latent_fc_layers
        self.dec_interaction = dec_interaction
        self.T_layer_norm = T_layer_norm
        self.dec_type = dec_type

        self.map_noise = nn.Sequential(
            nn.Linear(noise_size, embed_size),
            nn.LayerNorm(embed_size)
        )

        if dec_interaction == 'concat':
            self.interaction_fc = self.concat_fn()
        elif dec_interaction == 'add':
            self.interaction_fc = self.add_fn()
        else:
            raise ValueError

        if T_layer_norm:
            self.norm_prev = nn.LayerNorm(embed_size)
            self.norm_post = nn.LayerNorm(embed_size)
        
        if dec_interaction == 'concat':
            self.recovery_fc = nn.Sequential(
                nn.Linear(embed_size*2, embed_size),
                
            )

        dec_fc_layers = []
        for layer_i in range(num_dec_fc_layers):
            dec_fc_layers.append(
                nn.Sequential(
                    ResBlock(embed_size + content_size, embed_size + content_size, afn='relu', nfn=None),
                    nn.LayerNorm(embed_size + content_size)
                )
            )
        self.dec_fc_layers = nn.Sequential(*dec_fc_layers)

        if self.dec_type == 'lstm':
            self.h_fc = nn.Linear(embed_size+content_size, dec_hidden_size)
            self.h_agn = nn.Tanh()
            self.c_fc = nn.Linear(embed_size+content_size, dec_hidden_size)
        elif self.dec_type == 'gru':
            self.h_fc = nn.Linear(embed_size+content_size, dec_hidden_size)
            self.h_agn = nn.Tanh()
        else:
            raise ValueError

    def concat_fn(self):
        return nn.Sequential(
            nn.Linear(self.embed_size*2, self.embed_size),
            nn.ReLU(),
            nn.LayerNorm(self.embed_size),
            nn.Linear(self.embed_size, self.embed_size)
        )
    
    def add_fn(self):
        middle_layers=[]
        middle_layers.append(
            nn.Sequential(
            ResBlock(self.embed_size*2, self.embed_size, afn='relu', nfn=None),
            nn.LayerNorm(self.embed_size))
        )

        for layer_i in range(self.num_latent_fc_layers-1):
            middle_layers.append(
                nn.Sequential(
                ResBlock(self.embed_size, self.embed_size, afn='relu', nfn=None),
                nn.LayerNorm(self.embed_size))
            )
        middle_layers.append(nn.Linear(self.embed_size, self.embed_size))
        return nn.Sequential(*middle_layers)

    def forward(self, z, embed, mode='trans'):
        
        prev_style = embed[:, self.content_size:]
        prev_content = embed[:, :self.content_size]

        if mode == 'trans' or mode == 'zero':
            
            
            z = self.map_noise(z)
            
            concat = torch.cat([z, prev_style], dim=1)
            style_dis = self.interaction_fc(concat)

            
            if self.T_layer_norm:
                style_dis = self.norm_prev(style_dis)
            
            if self.dec_interaction == 'add':
                
                style_new = style_dis + prev_style
            elif self.dec_interaction == 'concat':
                
                style_new = torch.cat([style_dis, prev_style], dim=1)
                style_new = self.recovery_fc(style_new)
            
            if self.T_layer_norm:
                style_new = self.norm_post(style_new)
            
            
            if self.content_size>0:
                new_code = torch.cat([prev_content, style_new], dim=1)
            else:
                new_code = style_new
        
            new_code = self.dec_fc_layers(new_code)
            
            h0 = self.h_fc(new_code)
            h0 = self.h_agn(h0)
            h0 = h0.unsqueeze(0)

            if self.dec_type == 'lstm':
                c0 = self.c_fc(new_code)
                c0 = c0.unsqueeze(0)
                return (h0, c0), style_new
            elif self.dec_type == 'gru':
                return h0, style_new
            else:
                raise ValueError

        elif mode == 'recon':
            if self.content_size > 0:
                new_code = torch.cat([prev_content, prev_style], dim=1)
            else:
                new_code = prev_style
            new_code = self.dec_fc_layers(new_code)

            h0=self.h_fc(new_code)
            h0=self.h_agn(h0)
            h0=h0.unsqueeze(0)
            
            if self.dec_type == 'lstm':
                c0=self.c_fc(new_code)
                c0=c0.unsqueeze(0)
                return (h0, c0), prev_style
            elif self.dec_type == 'gru':
                return h0, prev_style
            else:
                raise ValueError
            
        else:
            raise ValueError

class AudioTranslatorWrapper(nn.Module):
    def __init__(self,
        C_in,
        C_out,
        kernel_size,
    ):
        super(AudioTranslatorWrapper, self).__init__()
        self.conv_layer = SeqTranslator1D(
            C_in=C_in,
            C_out=C_out,
            kernel_size=kernel_size,
            stride=1,
            min_layers_num=3
        )
    
    def forward(self, x):
        return self.conv_layer(x)

class Generator(nn.Module):
    def __init__(self,
        pose_dim,
        embed_dim,
        content_dim,
        noise_dim,
        aud_dim,
        num_steps,
        seq_dec_hidden_size,
        seq_dec_num_layers,
        latent_enc_fc_size,
        latent_enc_num_layers,
        latent_dec_num_layers,
        aud_kernel_size,
        residual = True,
        local_bn = False,
        training=True,
        aud_decoding=True,
        recon_input=True,
        T_layer_norm=False,
        interaction='add',
        rnn_cell = 'gru',
        graph_type = 'part',
        share_weights=False
    ):
        super(Generator, self).__init__()
        self.training = training
        self.rnn_cell = rnn_cell
        self.noise_dim = noise_dim
        self.aud_decoding = aud_decoding
        self.recon_input = recon_input
        print("warning: layer configs for graph seq encoder are hard coded")
        if graph_type == 'part':
            A = np.stack([self_graph, hand_graph, arm_graph, body_graph])
        elif graph_type == 'whole':
            A = np.stack([self_graph, whole_graph])
        elif graph_type == 'part_class':
            A = np.stack([self_graph, hand_graph, arm_graph, body_graph, hand_class_graph, arm_class_graph])
        elif graph_type == 'part_class_share_weights':
            raise NotImplementedError


        self.seq_enc = SeqEncoderWrapper(
            A = A,
            T = num_steps,
            embed_size=embed_dim,
            layer_configs=[(2, 32, 5), 
                       (32, 32, 5), 
                       (32, 64, 5), 
                       (64, 64, 5), 
                       (64, 128, 5), 
                       (128, 128, 5), 
                       (128, 256, 5), 
                       (256, 256, 5), 
                       (256, 128, 5)],
            residual=residual,
            local_bn=local_bn,
            share_weights = share_weights
        )

        self.seq_dec = SeqDecoderWrapper(
            hidden_size=seq_dec_hidden_size,
            out_size=pose_dim,
            in_size = pose_dim,
            num_steps=num_steps,
            num_layers=seq_dec_num_layers,
            rnn_cell=rnn_cell
        )
        self.latent_enc = LatentEncoderWrapper(
            embed_size=embed_dim,
            content_size=content_dim,
            fc_size=latent_enc_fc_size,
            noise_size=noise_dim,
            num_latent_fc_layers=latent_enc_num_layers,
            interaction_method=interaction
        )
        self.latent_dec = LatentDecoderWrapper(
            embed_size=embed_dim,
            content_size=content_dim,
            noise_size=noise_dim,
            num_latent_fc_layers=latent_enc_num_layers,
            num_dec_fc_layers=latent_dec_num_layers,
            dec_interaction=interaction,
            dec_hidden_size=seq_dec_hidden_size,
            T_layer_norm=T_layer_norm,
            dec_type=rnn_cell
        )
        self.aud_translator = AudioTranslatorWrapper(
            C_in=aud_dim,
            C_out=pose_dim,
            kernel_size=aud_kernel_size
        )

    def forward(self, aud, pre_poses, gt_poses=None, mode='trans'):
        B, _, T = aud.shape
        if self.training:
            assert gt_poses is not None
            his_feat = self.seq_enc(pre_poses)
            fut_feat = self.seq_enc(gt_poses)
            mu, var, z = self.latent_enc(his_feat, fut_feat)
            if mode == 'trans':
                pass
            elif mode == 'zero':
                z = torch.zeros_like(z).to(aud.device)
            elif mode == 'recon':
                raise ValueError("attemp to use recon in the main forwarding, where mode should be trans or zero")
            else:
                raise ValueError
            hidden, new_style = self.latent_dec(z, his_feat, mode=mode)
            if len(new_style.shape) == 2:
                new_style = new_style.unsqueeze(1)
            # frame_0 = new_style

            frame_0 = pre_poses[:, :, -1:]#B,C,T

            assert frame_0.shape[0] == B
            
            pred_poses = self.seq_dec(hidden, frame_0)
            
            if self.recon_input:
                
                
                hidden_his, new_style = self.latent_dec(None, his_feat, mode='recon')
                if len(new_style.shape) == 2:
                    new_style = new_style.unsqueeze(1)
                frame_0_his = pre_poses[:, :, 0:1]
                # frame_0_his = new_style
                
                recon_poses_his = self.seq_dec(hidden_his, frame_0_his)

                hidden_fut, new_style = self.latent_dec(None, fut_feat, mode='recon')
                if len(new_style.shape) == 2:
                    new_style = new_style.unsqueeze(1)
                # frame_0_fut = new_style
                frame_0_fut = pre_poses[:, :, -1:]
                
                recon_poses_fut = self.seq_dec(hidden_fut, frame_0_fut)

                recon_poses = (recon_poses_his, recon_poses_fut)
            else:
                recon_poses = None

            
            if mode == 'zero' and self.aud_decoding:
                dynamics = self.aud_translator(aud)
                pred_poses = pred_poses + dynamics
            else:
                dynamics = None

            return pred_poses, recon_poses, mu, var, dynamics
        else:
            his_feat = self.seq_enc(pre_poses)

            if mode == 'trans':
                z = torch.randn([B, self.noise_dim]).to(aud.device)
            elif mode == 'zero':
                z = torch.zeros([B, self.noise_dim]).to(aud.device)
            elif mode == 'recon':
                z = None
            
            hidden, new_style = self.latent_dec(z, his_feat)
            if len(new_style.shape) == 2:
                new_style = new_style.unsqueeze(1)
            # frame_0 = new_style
            frame_0 = pre_poses[:, :, -1:]
            
            pred_poses = self.seq_dec(hidden, frame_0)

            if mode == 'zero' and self.aud_decoding:
                dynamics = self.aud_translator(aud)
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
            embed_dim=self.args.embed_dim,
            content_dim=self.args.content_dim,
            noise_dim=self.args.noise_dim,
            aud_dim=self.args.aud_dim,
            num_steps=self.args.generate_length,
            seq_dec_hidden_size=self.args.seq_dec_hidden_size,
            seq_dec_num_layers=self.args.seq_dec_num_layers,
            latent_enc_fc_size=self.args.latent_enc_fc_size,
            latent_enc_num_layers=self.args.latent_enc_num_layers,
            latent_dec_num_layers=self.args.latent_dec_num_layers,
            aud_kernel_size=self.args.aud_kernel_size,
            residual=self.args.residual,
            local_bn=self.args.local_bn,
            recon_input=self.args.recon_input,
            training=(not self.args.infer),
            aud_decoding=self.args.aud_decoding,
            T_layer_norm=self.args.T_layer_norm,
            interaction=self.args.interaction,
            rnn_cell=self.args.rnn_cell,
            graph_type=self.args.graph_type,
            share_weights = self.args.share_weights
        ).to(self.device)

        self.discriminator = None

        self.rec_loss = KeypointLoss().to(self.device)
        self.reg_loss = KeypointLoss().to(self.device)
        self.kl_loss = KLLoss(kl_tolerance=self.args.kl_tolerance).to(self.device)
        self.vel_loss = VelocityLoss(velocity_length=self.args.velocity_length).to(self.device)
        self.l2reg_loss = L2RegLoss().to(self.device)
        self.r_loss = AudioLoss().to(self.device)
        
        self.global_step = 0

        print("using v1 graph")

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
            
            curr_kl_weight=(self.args.kl_loss_weight - (self.args.kl_loss_weight - self.args.kl_start_weight)*(self.args.kl_decay_rate)**self.global_step)

            total_loss = self.args.keypoint_loss_weight * recon_loss + \
                        self.args.recon_input_weight * reg_loss + \
                            curr_kl_weight * vae_loss + \
                                self.args.vel_loss_weight * vel_loss + \
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
            if dynamics is not None:
                r_loss = self.r_loss(dynamics, gt_poses)
            else:
                r_loss = 0

            total_loss = self.args.keypoint_loss_weight * recon_loss + \
                        self.args.recon_input_weight * reg_loss + \
                            self.args.zero_loss_weight * vae_loss + \
                                self.args.vel_loss_weight * vel_loss + \
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
        aud_feat = get_melspec(aud_fn).transpose(1, 0)

        for step in range(len(code_seq)):
            aud_feat_step = aud_feat[:, step*generate_length: (step+1)*generate_length]
            if aud_feat_step.shape[-1] < generate_length:
                aud_feat_step = np.pad(aud_feat_step, [[0, 0], [0, generate_length-aud_feat_step.shape[-1]]], mode='constant')
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


if __name__ == '__main__':
    test_model = Generator(
        pose_dim=108,
        embed_dim=512,
        content_dim=256,
        noise_dim=256,
        aud_dim=64,
        num_steps=25,
        seq_enc_hidden_size=512,
        seq_enc_num_layers=1,
        seq_dec_hidden_size=512,
        seq_dec_num_layers=1,
        latent_enc_fc_size=512,
        latent_enc_num_layers=3,
        latent_dec_num_layers=3,
        aud_kernel_size=7,
        recon_input=True,
        training=True,
        aud_decoding=True,
        T_layer_norm=True,
        interaction='add',
        rnn_cell='lstm',
        bidirectional=False
    )
    
    from utils import get_parameter_size
    total_num, trainable_num = get_parameter_size(test_model)
    print(total_num, trainable_num)

    test_aud = torch.randn([10, 64, 25])
    test_poses = torch.randn([10, 108, 25])
    test_gt_poses = torch.randn([10, 108, 25])
    test_mode = 'trans'

    output, recon_poses, mu, var = test_model(test_aud, test_poses, test_gt_poses, test_mode)

    print(output.shape)
    print(recon_poses[0].shape, recon_poses[1].shape)
