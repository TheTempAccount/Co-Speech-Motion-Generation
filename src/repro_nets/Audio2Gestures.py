'''
modified from https://github.com/JingLi513/Audio2Gestures.git
'''
import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random

from repro_nets.module import ConvNet
from nets.base import TrainWrapperBaseClass
from nets.utils import denormalize
from data_utils import get_mfcc, get_melspec, get_mfcc_old, get_mfcc_psf

def init(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class VAE(nn.Module):
    def __init__(self,
        freqbasis,
        feat_in_time_domain
    ) -> None:
        super(VAE, self).__init__()
        self.global_step = 0
        #these two args seem to be never used
        self.freqbasis = freqbasis
        self.feat_in_time_domain = feat_in_time_domain 

    def reparameterize(cls, mu, logvar):
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def kl_scheduler(self):
        return max((self.global_step // 10) % 10000 * 0.0001, 0.0001)

    def kl_divergence(cls, mu, var):
        return torch.mean(-0.5 * torch.sum(1 + var - mu.pow(2) - var.exp(), dim=2,))

    def step(self):
        self.global_step += 1

class Audio_Enc(VAE):
    def __init__(self,
        freqbasis,
        feat_in_time_domain,
        audio_size,
        dropout,
        audio_hidden_size,
        with_audio_share_vae,
        lambda_kl
    ):
        super(Audio_Enc, self).__init__(
            freqbasis,
            feat_in_time_domain
        )

        self.with_audio_share_vae = with_audio_share_vae
        self.lambda_kl = lambda_kl

        self.TCN = ConvNet(
            audio_size, [128, 128, 96, 96, 64], dropout=dropout
        )
        self.share_mean = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, audio_hidden_size),
        )
        self.share_var = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, audio_hidden_size),
        )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        ret:
            (B, T, audio_hidden_size)
        """
        inputs = inputs.permute(0, 2, 1)# the data format in this repo is (B, C, T)
        output = self.TCN(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        if self.with_audio_share_vae:
            self.z_share_mu = self.share_mean(output)
            self.z_share_var = self.share_var(output)
            z_share = self.reparameterize(self.z_share_mu, self.z_share_var)
        else:
            z_share = self.share_mean(output)
        return z_share

    def get_loss_dict(self):
        loss_dict = {}
        if self.with_audio_share_vae:
            loss_dict.update(
                {
                    "KL/audio_share": self.kl_divergence(
                        self.z_share_mu, self.z_share_var
                    )
                    * self.lambda_kl
                    * self.kl_scheduler(),
                }
            )
        self.step()
        return loss_dict

class Motion_Enc(VAE):
    def __init__(self,
        freqbasis,
        feat_in_time_domain,
        joint_num,
        dropout,
        pose_hidden_size,
        with_motion_share_vae,
        with_motion_spec_vae,
        lambda_kl
    ):
        super(Motion_Enc, self).__init__(
            freqbasis,
            feat_in_time_domain,
        )
        self.with_motion_share_vae = with_motion_share_vae
        self.with_motion_spec_vae = with_motion_spec_vae
        self.lambda_kl = lambda_kl
        joint_repr_dim = 2
        input_channel = joint_num * joint_repr_dim
        self.TCN = ConvNet(
            input_channel, [256, 256, 128, 128, 64], dropout=dropout,
        )
        self.share_linear = nn.Linear(64, 32)
        self.spec_linear = nn.Linear(64, 32)
        self.share_mean = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, pose_hidden_size),
        )
        self.share_var = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, pose_hidden_size),
        )
        self.spec_mean = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, pose_hidden_size),
        )
        self.spec_var = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, pose_hidden_size),
        )
    
    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        ret:  
            z_share: (B, T, pose_hidden_size)
            z_specific: (B, T, pose_hidden_size)
        """
        inputs = inputs.permute(0, 2, 1)# the data format in this repo is (B, C, T)
        output = self.TCN(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        share_output = self.share_linear(output)
        spec_output = self.spec_linear(output)

        if self.with_motion_share_vae:
            self.z_share_mu = self.share_mean(share_output)
            self.z_share_var = self.share_var(share_output)
            z_share = self.reparameterize(self.z_share_mu, self.z_share_var)
        else:
            z_share = self.share_mean(share_output)
        if self.with_motion_spec_vae:
            self.z_spec_mu = self.spec_mean(spec_output)
            self.z_spec_var = self.spec_var(spec_output)
            z_specific = self.reparameterize(self.z_spec_mu, self.z_spec_var)

        else:
            z_specific = self.spec_mean(spec_output)

        return z_share, z_specific

    def get_loss_dict(self):
        loss_dict = {}
        if self.with_motion_share_vae:
            loss_dict.update(
                {
                    "KL/motion_share": self.kl_divergence(
                        self.z_share_mu, self.z_share_var
                    )
                    * self.lambda_kl
                    * self.kl_scheduler(),
                }
            )
        if self.with_motion_spec_vae:
            loss_dict.update(
                {
                    "KL/motion_spec": self.kl_divergence(
                        self.z_spec_mu, self.z_spec_var
                    )
                    * self.lambda_kl
                    * self.kl_scheduler(),
                }
            )
        self.step()
        return loss_dict

class Motion_Dec(VAE):
    def __init__(self,
        freqbasis,
        feat_in_time_domain,
        joint_num,
        hidden_size,
        dropout
    ):
        super(Motion_Dec, self).__init__(
            freqbasis,
            feat_in_time_domain,
        )
        
        joint_repr_dim = 2
       
        output_dim = joint_repr_dim * joint_num

        self.TCN = ConvNet(
            hidden_size, [64, 128, 128, 256, 256,], dropout=dropout,
        )
        self.pose_g = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, output_dim),
        )

    def forward(self, share_feature: torch.Tensor, spec_feature: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        ret:
            (B, C, T)
        """
        #注意这里输入是(B, T, C)的code
        output = torch.cat((share_feature, spec_feature), dim=2)
        output = self.TCN(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.pose_g(output)

        return output.permute(0, 2, 1)#(B, T, C)->(B, C, T)

    def inference(self, z_share, z_spec):
        z = torch.cat((z_share, z_spec), dim=2)
        output = self.TCN(z.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.pose_g(output)
        return output.permute(0, 2, 1)

class MappingNet(VAE):
    def __init__(self,
        freqbasis,
        feat_in_time_domain,
        pose_hidden_size,
        with_mapping_net_vae,
        lambda_kl
    ):
        super(MappingNet, self).__init__(
            freqbasis,
            feat_in_time_domain,
        )
        self.with_mapping_net_vae = with_mapping_net_vae
        self.lambda_kl = lambda_kl
        hidden_size = pose_hidden_size
        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )
        self.spec_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.spec_var = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, inputs: torch.Tensor):
        '''
        inputs: (B, T, C)的latent code?
        '''
        output = self.net(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        if self.with_mapping_net_vae:
            self.z_spec_mu = self.spec_mean(output)
            self.z_spec_var = self.spec_var(output)
            z_specific = self.reparameterize(self.z_spec_mu, self.z_spec_var)
        else:
            z_specific = self.spec_mean(output)
        return z_specific

    def get_loss_dict(self):
        loss_dict = {}
        if self.with_mapping_net_vae:
            loss_dict.update(
                {
                    "KL/Mapping": self.kl_divergence(self.z_spec_mu, self.z_spec_var)
                    * self.lambda_kl
                    * self.kl_scheduler(),
                }
            )
        self.step()
        return loss_dict

class Motion_Process(nn.Module):
    def __init__(self) -> None:
        super(Motion_Process, self).__init__()

    def encode_motion(self, motion: torch.Tensor):
        return motion

    def decode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return motion

    def calculate_pos(self, motion: torch.Tensor) -> torch.Tensor:
        return motion

    def calculate_joint_speed(self, pos: torch.Tensor) -> torch.Tensor:
        assert pos.shape[1] == 108
        return pos[:, :, 1:] - pos[:, :, :-1]

class Process_S2G_Motion(Motion_Process):
    def __init__(self, 
            mean, 
            std
        ) -> None:
        super(Process_S2G_Motion, self).__init__()
        self.register_buffer(
            "mean",
            torch.FloatTensor(mean),
        )
        self.register_buffer(
            "std",
            torch.FloatTensor(
                std
                + np.finfo(float).eps
            ),
        )

    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        '''
        motion: (B, C, T)
        '''
        assert motion.shape[1] == 108
        motion = motion.permute(0, 2, 1) #(B, T, C)
        B, T = motion.shape[:2]
        motion = motion.view(B, T, -1, 2)
        assert len(motion.shape) == 4 and motion.shape[2] == 54
    
        motion -= motion[:, :, :, 1:2]
        motion = motion.reshape(B, T, -1)
        motion = (motion - self.mean) / self.std
        motion = motion.reshape(B, T, 54 * 2).permute(0, 2, 1)
        return motion

    def decode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        assert motion.shape[1] == 108
        motion = motion.permute(0, 2, 1)
        motion = motion * self.std + self.mean
        motion = motion.permute(0, 2, 1)
        return motion

class Generator(nn.Module):
    def __init__(self,
        audio_size,
        joint_num,
        hidden_size,
        audio_hidden_size,
        pose_hidden_size,
        with_code_constrain,
        with_mapping_net,
        with_cyc,
        with_ds,
        with_translation,
        with_audio_share_vae,
        with_motion_share_vae,
        with_motion_spec_vae,
        with_mapping_net_vae,
        using_mspec_stat,
        using_2D_data,
        seq_len,
        lambda_kl,
        dropout,
        device,
        mean,
        std
    ) -> None:
        super().__init__()
        self.using_mspec_stat = using_mspec_stat
        self.with_mapping_net = with_mapping_net
        self.seq_len = seq_len
        self.device = device

        self.net_G = nn.ModuleDict(
            {
                "audio_enc": Audio_Enc(
                    freqbasis=None,
                    feat_in_time_domain=None,
                    audio_size = audio_size,
                    dropout=dropout,
                    audio_hidden_size=audio_hidden_size,
                    with_audio_share_vae=with_audio_share_vae,
                    lambda_kl=lambda_kl
                ),
                "motion_enc": Motion_Enc(
                    freqbasis=None,
                    feat_in_time_domain=None,
                    joint_num=joint_num,
                    dropout=dropout,
                    pose_hidden_size=pose_hidden_size,
                    with_motion_share_vae=with_motion_share_vae,
                    with_motion_spec_vae=with_motion_spec_vae,
                    lambda_kl=lambda_kl
                ),
                "motion_dec": Motion_Dec(
                    freqbasis=None,
                    feat_in_time_domain=None,
                    joint_num=joint_num,
                    hidden_size=hidden_size,
                    dropout=dropout
                ),
                "mapping_net": MappingNet(
                    freqbasis=None,
                    feat_in_time_domain=None,
                    pose_hidden_size=pose_hidden_size,
                    with_mapping_net_vae=with_mapping_net_vae,
                    lambda_kl=lambda_kl
                )
            }
        )
        self.net_G.apply(init)

        self.motion_processor = Process_S2G_Motion(
            mean=mean,
            std = std
        )

    def sampling(self, size=None, mean=None, var=None):
        if self.using_mspec_stat:
            normal = Normal(mean, var)
            z_x = normal.sample((self.seq_len,)).permute(1, 0, 2)
        else:
            z_x = torch.randn(size, device=self.device)
        if self.with_mapping_net:
            z_x = self.net_G["mapping_net"](z_x)
        return z_x

    def forward(self, aud, pre_poses, gt_poses=None):
        '''
        aud: (B, C, T)
        pre_poses: None, not uses
        gt_poses: (B, C, T)
        '''
        self.z_audio_share = self.net_G['audio_enc'](aud)
        (self.z_motion_share, self.z_motion_specific,) = self.net_G["motion_enc"](
            gt_poses
        )
        recon_m = self.net_G["motion_dec"](self.z_motion_share, self.z_motion_specific)
        a2m = self.net_G["motion_dec"](self.z_audio_share, self.z_motion_specific)
        self.z_x = self.sampling(
            size=self.z_motion_specific.shape,
            mean=self.z_motion_specific.mean(dim=(1,)),
            var=self.z_motion_specific.std(dim=(1,)),
        )
        z_x2 = self.sampling(
            size=self.z_motion_specific.shape,
            mean=self.z_motion_specific.mean(dim=(1,)),
            var=self.z_motion_specific.std(dim=(1,)),
        )
        a2x = self.net_G["motion_dec"](self.z_audio_share, self.z_x)
        a2x2 = self.net_G["motion_dec"](self.z_audio_share, z_x2)
        
        #skip denormalize and normalize
        a2x = self.motion_processor.decode_motion(a2x)
        (_, self.z_a2x_spec) = self.net_G["motion_enc"](
                self.motion_processor.encode_motion(a2x)
            )
        
        return recon_m, a2m, a2x, a2x2, self.z_audio_share, self.z_motion_share, self.z_motion_specific, self.z_x, self.z_a2x_spec
    
    def inference(self, audios, motions):
        '''
        B, C, T
        '''
        # if self.seq_len > 0:
        #     seq_len = min(audios.shape[2], self.seq_len)
        # else:
        #     seq_len = audios.shape[2]
        seq_len = audios.shape[2]
        
        if motions is None:
            if self.using_mspec_stat:#training的时候是需要使用的
                '''
                '''
                raise NotImplementedError
            
        z_audio_share = self.net_G['audio_enc'](audios[:, :, :seq_len])
        if motions is None:
            if self.using_mspec_stat:
                raise NotImplementedError
                idx = random.randint(0, means.shape[0] - 1)
                z_motion_spec = self.sampling(
                    mean=means[idx : idx + 1], var=vars[idx : idx + 1],
                )
            else:
                '''
                采样一个
                '''
                z_motion_spec = self.sampling(size=z_audio_share.shape)
        else:
            '''
            否则使用motion的encoding
            '''
            _, z_motion_spec = self.net_G["motion_enc"](motions[:, :, :seq_len])
        pred_motions = self.net_G["motion_dec"].inference(z_audio_share, z_motion_spec)

        return pred_motions


class TrainWrapper(TrainWrapperBaseClass):
    def __init__(self, 
        args, 
        config,
        mean,
        std
    ) -> None:
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0

        torch.backends.cudnn.deterministic = False #faster, 'cause the dilation conv

        self.generator = Generator(
            audio_size = config.Data.aud.aud_feat_dim,
            joint_num = config.Data.pose.pose_dim // 2,
            hidden_size = config.Model.hidden_size,
            audio_hidden_size = config.Model.audio_hidden_size,
            pose_hidden_size = config.Model.pose_hidden_size,
            with_code_constrain = config.Train.with_code_constrain,
            with_mapping_net = config.Model.with_mapping_net,
            with_cyc = config.Train.with_cyc,
            with_ds = config.Train.with_ds,
            with_translation = config.Train.with_translation,
            with_audio_share_vae = config.Train.with_audio_share_vae,
            with_motion_share_vae = config.Train.with_motion_share_vae,
            with_motion_spec_vae = config.Train.with_motion_spec_vae,
            with_mapping_net_vae = config.Train.with_mapping_net_vae,
            using_mspec_stat = config.Train.using_mspec_stat,
            using_2D_data = config.Data.using_2D_data,
            seq_len = config.Data.seq_len,
            lambda_kl = config.Train.weights.lambda_kl,
            dropout = config.Model.dropout,
            device = self.device,
            mean=mean,
            std=std
        ).to(self.device)

        self.init_lambda_ds = config.Train.weights.lambda_ds
        self.lambda_ds = self.init_lambda_ds

        self.discriminator = None

        super().__init__(args, config)
    
    def __call__(self, bat):
        assert (not self.args.infer), "infer mode"
        self.generator.train()
        self.global_step += 1
        #TODO: check this
        if self.lambda_ds > 0:
            self.lambda_ds -= self.init_lambda_ds / 500000

        aud, gt_poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)
        audios = aud
        motion = gt_poses

        src_motion = motion.clone()
        tgt_motion = motion.clone()

        recon_m, a2m, a2x, a2x2, self.z_audio_share, self.z_motion_share, self.z_motion_specific, self.z_x, self.z_a2x_spec = self.generator(
            aud = audios,
            pre_poses = None,
            gt_poses = src_motion
        )  

        total_loss, loss_dict = self.get_loss(
            tgt_motion, recon_m, a2m, a2x, a2x2
        )


        self.generator.zero_grad()
        self.generator_optimizer.zero_grad()
        total_loss.backward()
        self.generator_optimizer.step()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5.0)

        loss_dict['grad'] = grad_norm
        return total_loss, loss_dict

    def get_loss(self,
        tgt_p,
        recon_p,
        a2m_p,
        a2x_p,
        a2x2_p,
    ):
        tgt_p = self.generator.motion_processor.decode_motion(tgt_p)
        recon_p = self.generator.motion_processor.decode_motion(recon_p)
        a2m_p = self.generator.motion_processor.decode_motion(a2m_p)
        a2x_p = self.generator.motion_processor.decode_motion(a2x_p)
        a2x2_p = self.generator.motion_processor.decode_motion(a2x2_p)

        tgt_s = self.generator.motion_processor.calculate_joint_speed(tgt_p)
        recon_s = self.generator.motion_processor.calculate_joint_speed(recon_p)
        a2m_s = self.generator.motion_processor.calculate_joint_speed(a2m_p)
        a2x_s = self.generator.motion_processor.calculate_joint_speed(a2x_p)
        joint_distance = torch.abs(a2x_p - tgt_p)
        loss_dict = {
            "pos/recon_position": F.l1_loss(recon_p, tgt_p) * self.config.Train.weights.lambda_pose,
            "speed/recon_speed": F.l1_loss(recon_s, tgt_s) * self.config.Train.weights.lambda_speed,
            "pos/audio2position": F.l1_loss(a2m_p, tgt_p) * self.config.Train.weights.lambda_pose,
            "speed/audio2speed": F.l1_loss(a2m_s, tgt_s) * self.config.Train.weights.lambda_speed,
            "pos/audio2position_x": joint_distance[
                joint_distance > self.config.Train.weights.tolerance
            ].mean()
            * self.config.Train.weights.lambda_pose,
            "speed/audio2speed_x": F.l1_loss(a2x_s, tgt_s) * self.config.Train.weights.lambda_xspeed,
        }

        if self.config.Train.with_code_constrain:
            loss_dict.update(
                {
                    "code/share_code_constrain": F.l1_loss(
                        self.z_audio_share, self.z_motion_share
                    )
                    * self.config.Train.weights.lambda_code,
                }
            )
        if self.config.Train.with_cyc:
            loss_dict.update(
                {
                    "code/cyc": F.l1_loss(self.z_a2x_spec, self.z_x)
                    * self.config.Train.weights.lambda_cyc
                }
            )
        if self.config.Train.with_ds:
            loss_dict.update(
                {
                    "pos/diverse": -F.l1_loss(a2x_p, a2x2_p.detach())
                    * self.config.Train.weights.lambda_ds,
                }
            )
        
        loss_dict.update(self.generator.net_G["audio_enc"].get_loss_dict())
        loss_dict.update(self.generator.net_G["motion_enc"].get_loss_dict())
        loss_dict.update(self.generator.net_G["mapping_net"].get_loss_dict())
        total_loss = torch.stack(list(loss_dict.values())).sum()

        return total_loss, loss_dict

    def infer_on_audio(self, aud_fn, initial_pose=None, norm_stats=None, **kwargs):
        '''
        initial_pose: used to specify batch_size
        '''
        output = []

        assert self.args.infer, "train mode"
        self.generator.eval()

        if self.config.Data.pose.normalization:
            assert norm_stats is not None
            data_mean = norm_stats[0]
            data_std = norm_stats[1]
        
        if self.config.Data.aud.feat_method == 'mel_spec':
            aud_feat = get_melspec(aud_fn).transpose(1, 0) #(C, T)
        elif self.config.Data.aud.feat_method == 'mfcc':
            aud_feat = get_mfcc_psf(aud_fn).transpose(1, 0)
        elif self.config.Data.aud.aud_feat_win_size is not None:
            aud_feat = get_mfcc_old(aud_fn)

        aud_feat = aud_feat[np.newaxis, :].repeat(initial_pose.shape[0], axis=0)
        aud_feat = torch.tensor(aud_feat, dtype = torch.float32).to(self.device)
        
        with torch.no_grad():
            pred_poses = self.generator.inference(aud_feat, motions=None)
        
        pred_poses = pred_poses.clone().cpu().detach().numpy().transpose(0, 2, 1) 
        pred_poses = denormalize(pred_poses, data_mean, data_std)
        print(pred_poses.shape)
        return pred_poses


if __name__ == "__main__":
#     parser = parse_args()
#     torch.backends.cudnn.deterministic = True
    
    audio_size = 64
    joint_num = 54
    hidden_size = 32
    audio_hidden_size = 16
    pose_hidden_size = 16
    with_code_constrain = False
    with_mapping_net = False
    with_cyc = False
    with_ds = False
    with_translation = False
    with_audio_share_vae = False
    with_motion_share_vae = False
    with_motion_spec_vae = False
    with_mapping_net_vae = False
    using_mspec_stat = False
    using_2D_data = False
    seq_len = 32
    lambda_kl = 1e-4
    dropout = 0
    
    
    test_model = Generator(
        audio_size = 64,
        joint_num = 54,
        hidden_size = 32,
        audio_hidden_size = 16,
        pose_hidden_size = 16,
        with_code_constrain = True,
        with_mapping_net = True,
        with_cyc = True,
        with_ds = True,
        with_translation = False,
        with_audio_share_vae = True,
        with_motion_share_vae = True,
        with_motion_spec_vae = True,
        with_mapping_net_vae = True,
        using_mspec_stat = True,
        using_2D_data = True,
        seq_len = 128,
        lambda_kl = 1e-4,
        dropout = 0,
        device=torch.device(0),
        mean = torch.zeros(108),
        std = torch.ones(108)
    ).cuda()
    aud = torch.randn([32, 64, 128]).cuda()
    poses = torch.randn([32, 108, 128]).cuda()

    begin = time.time()
    output = test_model(aud, None, poses)
    end = time.time()
    print(end-begin)
    # print(output.shape)
