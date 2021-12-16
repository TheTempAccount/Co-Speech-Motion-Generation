import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from nets.layers import *

class Generator(nn.Module):
    '''
    简单的LSTM-FC的形式，将pre_poses和audio的embedding叠加之后，作为输入。原文中是使用pre_poses初始化一个空序列的
    '''
    def __init__(self,
        audio_dim,
        embed_dim,
        pose_dim,
        hidden_size,
        dropout_p,
        z_size,
        num_layers
    ):  
        super().__init__()
        self.audio_encoder = AudioPoseEncoder1D(
            C_in=audio_dim,
            C_out=embed_dim,
            min_layer_nums=3
        )
        self.z_size = z_size
        self.hidden_size = hidden_size
        
        in_size = embed_dim + pose_dim + z_size

        self.gru = AudioPoseEncoderRNN(
            C_in=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout_p)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(inplace = False),
            nn.Linear(self.hidden_size // 2, pose_dim)
        )

    def forward(self, in_audio, pre_seq):
        '''
        pre_seq: (B, C, T)
        in_audio: (B, C, T)
        '''
        audio_feat = self.audio_encoder(in_audio)
        in_data = torch.cat((pre_seq, audio_feat), dim=1)

        z_context = torch.randn(in_audio.shape[0], self.z_size, device = in_audio.device)
        repeated_z = z_context.unsqueeze(-1)
        repeated_z = repeated_z.repeat(1, 1, in_data.shape[-1])
        in_data = torch.cat((in_data, repeated_z), dim=1)
        in_data = in_data.to(torch.float32)

        output = self.gru(in_data, None)
        output = output[:, :self.hidden_size, :] + output[:, self.hidden_size:, :]
        output = output.permute(0, 2, 1)
        output = self.out(output)
        output = output.permute(0, 2, 1)

        return output, z_context

class ConvDiscriminator(nn.Module):
    '''
    conv-gru-fc，gru消除掉一个维度，fc消除掉第二个维度
    '''
    def __init__(self, 
        input_size,
        embed_size,
        hidden_size,
        num_rnn_layers
    ):
        super().__init__()
        self.input_size = input_size

        self.hidden_size = hidden_size
        self.pre_conv = AudioPoseEncoder1D(
            C_in=input_size,
            C_out=embed_size,
            min_layer_nums=3
        )

        self.gru=SeqEncoderRNN(
            hidden_size=hidden_size,
            in_size=embed_size,
            num_rnn_layers=num_rnn_layers,
            bidirectional=True
        )
        self.out = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        
        x = x.to(torch.float32)

        x = self.pre_conv(x)

        x = self.gru(x)

        x = self.out(x)

        return x

if __name__ == "__main__":
    test_model = ConvDiscriminator(
        input_size=108,
        embed_size=128,
        hidden_size=256,
        num_rnn_layers=1
    )

    dummy_input = torch.randn([64, 64, 43])
    dummy_pre_poses = torch.randn([64, 108, 43])
    output = test_model(dummy_pre_poses)
    print(output.shape)