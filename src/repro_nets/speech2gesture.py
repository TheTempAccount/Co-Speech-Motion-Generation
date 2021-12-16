import sys
import os
sys.path.append(os.getcwd())


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from nets import TrainWrapperBaseClass
from nets.layers import SeqEncoder1D
from losses import KeypointLoss, L1Loss, KLLoss
from data_utils.utils import get_melspec
from nets.utils import denormalize

""" from https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context.git """

class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    from https://github.com/mlperf/inference/blob/482f6a3beb7af2fb0bd2d91d6185d5e71c22c55f/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv1d_tf(nn.Conv1d):
    """
    Conv1d with the padding behavior from TF
    modified from https://github.com/mlperf/inference/blob/482f6a3beb7af2fb0bd2d91d6185d5e71c22c55f/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py
    """

    def __init__(self, *args, **kwargs):
        super(Conv1d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        if rows_odd:
            input = F.pad(input, [0, rows_odd])

        return F.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


def ConvNormRelu(in_channels, out_channels, type='1d', downsample=False, k=None, s=None, padding='SAME'):
    if k is None and s is None:
        if not downsample:
            k = 3
            s = 1
        else:
            k = 4
            s = 2

    if type == '1d':
        conv_block = Conv1d_tf(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
        norm_block = nn.BatchNorm1d(out_channels)
    elif type == '2d':
        conv_block = Conv2d_tf(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
        norm_block = nn.BatchNorm2d(out_channels)
    else:
        assert False

    return nn.Sequential(
        conv_block,
        norm_block,
        nn.LeakyReLU(0.2, True)
    )


class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetUp, self).__init__()
        self.conv = ConvNormRelu(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = torch.repeat_interleave(x1, 2, dim=2) #tf.keras.repeat_elements
        x1 = x1[:, :, :x2.shape[2]]  # to match dim
        x = x1 + x2  # it is different to the original UNET, but I stick to speech2gesture implementation
        x = self.conv(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_frames):
        super().__init__()
        self.n_frames = n_frames
        self.first_net = nn.Sequential(
            ConvNormRelu(1, 64, '2d', False),
            ConvNormRelu(64, 64, '2d', True),
            ConvNormRelu(64, 128, '2d', False),
            ConvNormRelu(128, 128, '2d', True),
            ConvNormRelu(128, 256, '2d', False),
            ConvNormRelu(256, 256, '2d', True),
            ConvNormRelu(256, 256, '2d', False),
            ConvNormRelu(256, 256, '2d', False, padding='VALID')
        )

        # self.make_1d = torch.nn.Upsample((n_frames, 1), mode='bilinear', align_corners=False)

        self.down1 = nn.Sequential(
            ConvNormRelu(256, 256, '1d', False),
            ConvNormRelu(256, 256, '1d', False)
        )
        self.down2 = ConvNormRelu(256, 256, '1d', True)
        self.down3 = ConvNormRelu(256, 256, '1d', True)
        self.down4 = ConvNormRelu(256, 256, '1d', True)
        self.down5 = ConvNormRelu(256, 256, '1d', True)
        self.down6 = ConvNormRelu(256, 256, '1d', True)
        self.up1 = UnetUp(256, 256)
        self.up2 = UnetUp(256, 256)
        self.up3 = UnetUp(256, 256)
        self.up4 = UnetUp(256, 256)
        self.up5 = UnetUp(256, 256)

    def forward(self, spectrogram, time_steps=None):
        spectrogram = spectrogram.unsqueeze(1)  # add channel dim
        # print(spectrogram.shape)
        spectrogram = spectrogram.float()

        if time_steps is None:
            time_steps = self.n_frames

        out = self.first_net(spectrogram)

        # 原始的tf代码应该是这样的
        # out = self.make_1d(out)
        out = torch.nn.functional.interpolate(out, size=(time_steps, 1), mode='bilinear')

        x1 = out.squeeze(3)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)

        return x


class Generator(nn.Module):
    '''
    通过n_poses的指定可以设置生成结果的fps。模型的结构是确定的
    TODO: mel_spec和MFCC，aud_feat的维度设定成了num_frames，但是原版的模型实现里面并不是这样的
    '''
    def __init__(self, n_poses, pose_dim, n_pre_poses, use_template=False, template_length=0):
        super().__init__()

        self.use_template = use_template
        self.template_length = template_length
        if self.use_template:
            assert template_length > 0 

        self.gen_length = n_poses

        self.audio_encoder = AudioEncoder(n_poses)#这里的这种写法会导致每一步只能生成固定长度的
        self.pre_pose_encoder = nn.Sequential(
            nn.Linear(n_pre_poses * pose_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            ConvNormRelu(256 + 16 + self.template_length, 256),
            ConvNormRelu(256, 256),
            ConvNormRelu(256, 256),
            ConvNormRelu(256, 256)
        )
        self.final_out = nn.Conv1d(256, pose_dim, 1, 1)

    def forward(self, in_spec, pre_poses, template=None, time_steps = None):
        if time_steps is not None:
            self.gen_length = time_steps
            
        if self.use_template:
            if template is None:
                template = torch.randn([in_spec.shape[0], self.template_length]).to(in_spec.device)
        
        #in_spec: (B, T, C). 在audio_encoder中
        audio_feat_seq = self.audio_encoder(in_spec, time_steps=time_steps)  # output (bs, feat_size, n_frames)
        pre_poses = pre_poses.reshape(pre_poses.shape[0], -1)
        pre_pose_feat = self.pre_pose_encoder(pre_poses)  # output (bs, 16)
        pre_pose_feat = pre_pose_feat.unsqueeze(2).repeat(1, 1, self.gen_length)

        if self.use_template:
            template_feat = template.unsqueeze(2).repeat(1, 1, self.gen_length)
            feat = torch.cat((audio_feat_seq, pre_pose_feat, template_feat), dim=1)
        else:
            feat = torch.cat((audio_feat_seq, pre_pose_feat), dim=1)

        out = self.decoder(feat)
        out = self.final_out(out)
        out = out.transpose(1, 2)  # to (batch, seq, dim)

        return out


class Discriminator(nn.Module):
    def __init__(self, pose_dim):
        super().__init__()
        self.net = nn.Sequential(
            Conv1d_tf(pose_dim, 64, kernel_size=4, stride=2, padding='SAME'),
            nn.LeakyReLU(0.2, True),
            ConvNormRelu(64, 128, '1d', True),
            ConvNormRelu(128, 256, '1d', k=4, s=1),
            Conv1d_tf(256, 1, kernel_size=4, stride=1, padding='SAME'),
        )

    def forward(self, x):
        # x = x[:, 1:] - x[:, :-1]  # 不知为何还在这里做这个，明明已经转化为pose differences了
        x = x.transpose(1, 2)  # to (batch, dim, seq)

        out = self.net(x)
        return out

class TrainWrapper(TrainWrapperBaseClass):
    def __init__(self, args) -> None:
        super().__init__(args)

        #定义模型和loss
        self.generator = Generator(
            n_poses = self.args.generate_length,
            pose_dim = self.args.pose_dim,
            n_pre_poses = self.args.pre_pose_length,
            use_template = self.args.use_template,
            template_length = self.args.template_length
        ).to(self.device)
        self.discriminator = Discriminator(
            pose_dim=self.args.pose_dim
        ).to(self.device)

        self.MSELoss = KeypointLoss().to(self.device)
        self.L1Loss = L1Loss().to(self.device)
        if self.args.use_template:
            self.KLLoss = KLLoss(kl_tolerance=self.args.kl_tolerance).to(self.device)
            self.pose_encoder = SeqEncoder1D(
                C_in=self.args.pose_dim,
                C_out=64,
                T_in=self.args.generate_length,
                
            )
            self.mu_fc = nn.Linear(64, self.args.template_length)
            self.var_fc = nn.Linear(64, self.args.template_length)

        self.init_optimizer()

    def init_optimizer(self) -> None:
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

    def __reparam(self, mu, log_var):
        std=torch.exp(0.5*log_var)
        eps=torch.randn_like(std, device=self.device)
        z=eps*std+mu
        return z

    def __call__(self, bat):
        assert (not self.args.infer), "infer mode"
        self.global_step += 1

        #poses是gt_poses, pre_poses是poses
        aud, gt_poses, pre_poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32), bat['pre_poses'].to(torch.float32).to(self.device)

        gt_conf = bat['conf'].to(self.device).to(torch.float32)

        if self.args.use_template:
            pose_enc = self.pose_encoder(gt_poses)
            mu = self.mu_fc(pose_enc)
            var = self.var_fc(pose_enc)
            template = self.__reparam(mu, var)#(B, D)
            kld_loss = self.KLLoss(mu, var)
        else:
            template = None
            kld_loss = 0

        aud = aud.permute(0, 2, 1)
        gt_poses = gt_poses.permute(0, 2, 1)
        pre_poses = pre_poses.permute(0, 2, 1)
        gt_conf = gt_conf.permute(0, 2, 1)


        if self.args.use_template:
            pred_poses = self.generator(
                in_spec = aud,#(B, C, T)->(B, T, C)
                pre_poses = pre_poses,
                template = template
            )

        #训练D
        D_loss, D_loss_dict = self.get_loss(
            pred_poses = pred_poses.detach(),
            gt_poses = gt_poses,
            mode='training_D',
            gt_conf=gt_conf)
        
        self.discriminator_optimizer.zero_grad()
        D_loss.backward()
        self.discriminator_optimizer.step()
        
        #训练G
        G_loss, G_loss_dict = self.get_loss(
            pred_poses = pred_poses,
            gt_poses = gt_poses,
            mode = 'training_G',
            gt_conf = gt_conf
        )
        self.generator_optimizer.zero_grad()
        G_loss += kld_loss
        G_loss.backward()
        self.generator_optimizer.step()

        total_loss = None
        loss_dict = {}
        if self.args.use_template:
            loss_dict['kld_loss'] = kld_loss
        for key in list(D_loss_dict.keys()) + list(G_loss_dict.keys()):
            loss_dict[key] = G_loss_dict.get(key, 0) + D_loss_dict.get(key, 0)
        
        return total_loss, loss_dict
    
    def get_loss(self,
        pred_poses,
        gt_poses,
        mode='training_G',
        gt_conf=None 
    ):
        loss_dict = {}
        target_motion = gt_poses[:, 1:]  - gt_poses[:, :-1]
        pred_motion = pred_poses[:, 1:] - pred_poses[:, :-1]

        if mode == 'training_D':
            dis_real = self.discriminator(target_motion)
            dis_fake = self.discriminator(pred_motion)
            dis_error = self.MSELoss(torch.ones_like(dis_real).to(self.device), dis_real) + self.MSELoss(torch.zeros_like(dis_fake).to(self.device), dis_fake)
            loss_dict['dis'] = dis_error

            return dis_error, loss_dict
        elif mode == 'training_G':
            l1_loss = self.L1Loss(pred_poses, gt_poses)
            dis_output = self.discriminator(pred_motion)
            gen_error = self.MSELoss(torch.ones_like(dis_output).to(self.device), dis_output)
            gen_loss = self.args.keypoint_loss_weight * l1_loss + self.args.gan_loss_weight * gen_error

            loss_dict['gen'] = gen_error
            loss_dict['l1_loss'] = l1_loss
            return gen_loss, loss_dict
        else:
            raise ValueError(mode)
 
    def state_dict(self):
        return super().state_dict()
    
    def parameters(self):
        return super().parameters()

    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)
    
    def infer_on_audio(self, aud_fn, initial_pose=None, norm_stats=None, **kwargs):
        '''
        initial_pose: (B, C, T)
        对于不同的aud_fn生成的长度不一样，一种选择是类似原始的speech2gesture，每个aud_fn重新声明模型，设定不同的生成长度，由于只在没有参数的上采样中涉及了生成长度，所以还是可以读取同一个ckpt的。另一种是类似trimodal的，分段然后smoothing的方式
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
        assert pre_length == initial_pose.shape[-1]
        pre_poses = initial_pose.permute(0, 2, 1).to(self.device).to(torch.float32)#(B, T, C)
        B=pre_poses.shape[0]
        

        #TODO: 暂时这里的T就是num-frames，注意原版的并不是
        aud_feat = get_melspec(aud_fn).transpose(1, 0)#(C, T)
        num_poses_to_generate = aud_feat.shape[-1]#(T)
        if False: #这里是trimodal的做法
            num_steps = math.ceil(num_poses_to_generate / generate_length) + 1
            generate_stride = generate_length - pre_length#每步生成的时候，相邻两步取特征的间隔

            for i in range(num_steps):
                step_start = i*generate_stride
                aud_feat_step = aud_feat[:, step_start:step_start+generate_length]#(C, T), T=generate_length
                if aud_feat_step.shape[-1] < generate_length:
                    aud_feat_step = np.pad(aud_feat_step, [[0, 0], [0, generate_length-aud_feat_step.shape[-1]]], mode='constant')
                aud_feat_step = aud_feat_step[np.newaxis, ...].repeat(B, axis=0)
                aud_feat_step = torch.tensor(aud_feat_step, dtype = torch.float32).to(self.device)

                with torch.no_grad():
                    aud_feat_step = aud_feat_step.permute(0, 2, 1)#(B, T,C)
                    pred_poses = self.generator(aud_feat_step, pre_poses)#(B, T, C)
                    pre_poses = pred_poses.detach().clone()[:, -pre_length:, :]#(B, pre_length, C)
                pred_poses = pred_poses.cpu().numpy()

                if len(output)>0:
                    last_poses = output[-1][:, -pre_length:]
                    output[-1] = output[-1][:, :-pre_length]

                    for j in range(pre_length):
                        n = pre_length
                        prev = last_poses[:, j]
                        next = pred_poses[:, j]
                        pred_poses[:, j] = prev * (n-j) / (n+1) + next * (j+1) / (n+1)
                output.append(pred_poses)

            output = np.concatenate(output, axis=1)
        else:
            aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
            aud_feat = torch.tensor(aud_feat, dtype = torch.float32).to(self.device)

            with torch.no_grad():
                aud_feat = aud_feat.permute(0, 2, 1)#(B, T,C)
                pred_poses = self.generator(aud_feat, pre_poses, time_steps = num_poses_to_generate)#(B, T, C)
                pred_poses = pred_poses.cpu().numpy()
            output = pred_poses
        if self.args.normalization:
            output = denormalize(output, data_mean, data_std)
        
        print(output.shape)
        return output


if __name__ == '__main__':
    # for model debugging
    # pose_dim = 16
    # generator = Generator(64, pose_dim, 4)#每次生成64帧
    # spec = torch.randn((4, 128, 64))#（B, T, C），64是mel_bins
    # pre_poses = torch.randn((4, 4, pose_dim))
    # generated = generator(spec, pre_poses)
    # print('spectrogram', spec.shape)
    # print('output', generated.shape)

    # discriminator = Discriminator(pose_dim)
    # out = discriminator(generated)
    # print('discrimination output', out.shape)

    from trainer.options import parse_args
    parser = parse_args()
    args = parser.parse_args(['--exp_name', '0', '--data_root','0','--speakers', '0', '--pre_pose_length', '4', '--generate_length', '64','--infer'])

    generator = TrainWrapper(args)

    # dummy_bat = {
    #     'aud_feat': torch.randn(64, 64, 64),
    #     'poses': torch.randn(64, 108, 64),
    #     'pre_poses': torch.randn(64, 108, 4),
    #     'conf': torch.randn(64, 108, 64)
    # }

    # for _ in range(10):
    #     _, loss_dict = generator(dummy_bat)
    #     for k in list(loss_dict.keys()):
    #         print(k, loss_dict[k])

    # print(generator.state_dict().keys())
    # print(type(generator.parameters()))

    aud_fn = '../sample_audio/jon.wav'
    initial_pose = torch.randn(64, 108, 4)
    norm_stats = (np.random.randn(108), np.random.randn(108))
    output = generator.infer_on_audio(aud_fn, initial_pose, norm_stats)

    print(output.shape) 

    


