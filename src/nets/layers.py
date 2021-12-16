from os import stat_result
from typing import ForwardRef
import torch
import torch.nn as nn
from torch.nn.modules import rnn

def get_log(x):
    log=0
    while x>1:
        if x%2 == 0:
            x = x // 2
            log += 1
        else:
            raise ValueError('x is not a power of 2')

    return log

class ConvNormRelu(nn.Module):
    '''
    (B,C_in,H,W) -> (B, C_out, H, W) 
    存在一些kernel size的选择会导致输出不是H/s
    '''
    def __init__(self, 
                    in_channels, 
                    out_channels,
                    type='1d', 
                    leaky=False,
                    downsample=False, 
                    kernel_size=None, 
                    stride=None,
                    padding=None, 
                    p=0, 
                    groups=1,
                    residual=False):
        '''
        简单的conv-bn-relu
        '''
        super(ConvNormRelu, self).__init__()
        self.residual = residual

        if kernel_size is None and stride is None:
            if not downsample:
                kernel_size = 3
                stride = 1
            else:
                kernel_size = 4 
                stride = 2
        
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(stride, tuple):
                padding = tuple(int((kernel_size - st)/2) for st in stride)
            elif isinstance(kernel_size, tuple) and isinstance(stride, int):
                padding = tuple(int((ks - stride)/2) for ks in kernel_size)
            elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
                padding = tuple(int((ks - st)/2) for ks, st in zip(kernel_size, stride))
            else:
                padding = int((kernel_size - stride)/2)

        
        if self.residual:
            if downsample:
                if type == '1d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )
                elif type == '2d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )
            else:
                if in_channels == out_channels:
                    self.residual_layer = nn.Identity()
                else:
                    if type == '1d':
                        self.residual_layer = nn.Sequential(
                            nn.Conv1d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        )
                    elif type == '2d':
                        self.residual_layer = nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        )
            

        in_channels = in_channels*groups
        out_channels = out_channels*groups
        if type == '1d':
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    groups=groups)
            self.norm = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(p=p)
        elif type == '2d':
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    groups=groups)
            self.norm = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(p=p)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        out = self.norm(self.dropout(self.conv(x)))
        if self.residual:
            residual = self.residual_layer(x)
            out += residual
        return self.relu(out)

class UNet1D(nn.Module):
    def __init__(self, 
                input_channels, 
                output_channels, 
                max_depth=5, 
                kernel_size=None, 
                stride=None, 
                p=0, 
                groups=1):
        super(UNet1D, self).__init__()
        self.pre_downsampling_conv = nn.ModuleList([])
        self.conv1 = nn.ModuleList([])
        self.conv2 = nn.ModuleList([])
        self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
        self.max_depth = max_depth
        self.groups = groups

        
        self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                    type='1d', leaky=True, downsample=False,
                                                    kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.pre_downsampling_conv.append(ConvNormRelu(output_channels, output_channels,
                                                    type='1d', leaky=True, downsample=False,
                                                    kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        
        for i in range(self.max_depth):
            self.conv1.append(ConvNormRelu(output_channels, output_channels,
                                            type='1d', leaky=True, downsample=True,
                                            kernel_size=kernel_size, stride=stride, p=p, groups=groups))

        for i in range(self.max_depth):
            self.conv2.append(ConvNormRelu(output_channels, output_channels,
                                            type='1d', leaky=True, downsample=False,
                                            kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    def forward(self, x):
        
        
        input_size = x.shape[-1]
        
        
        
        assert get_log(input_size) >= self.max_depth, 'num_frames must be a power of 2 and its power must be greater than max_depth'

        x = nn.Sequential(*self.pre_downsampling_conv)(x)

        residuals = []
        residuals.append(x)
        for i, conv1 in enumerate(self.conv1):
            x = conv1(x)
            if i < self.max_depth - 1:
                residuals.append(x)

        for i, conv2 in enumerate(self.conv2):
            x = self.upconv(x) + residuals[self.max_depth - i - 1]
            x = conv2(x)

        return x

class UNet2D(nn.Module):
    def __init__(self):
        super(UNet2D, self).__init__()
        raise NotImplementedError('2D Unet并不自然')



class AudioPoseEncoder1D(nn.Module):
    '''
    将audio的feat逐步提升
    (B, C, T) -> (B, C*2, T) -> ... -> (B, C_out, T)
    '''
    def __init__(self,
        C_in, 
        C_out,
        kernel_size=None,
        stride=None,
        min_layer_nums=None
    ):
        super(AudioPoseEncoder1D, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        conv_layers = nn.ModuleList([])
        cur_C = C_in
        num_layers=0
        while cur_C < self.C_out:
            conv_layers.append(ConvNormRelu(
                in_channels=cur_C,
                out_channels=cur_C*2,
                kernel_size=kernel_size,
                stride=stride
            ))
            cur_C *= 2
            num_layers += 1
        
        if (cur_C!=C_out) or (min_layer_nums is not None and num_layers < min_layer_nums):
            while (cur_C!=C_out) or num_layers < min_layer_nums:
                conv_layers.append(ConvNormRelu(
                    in_channels=cur_C,
                    out_channels=C_out,
                    kernel_size=kernel_size,
                    stride=stride
                ))
                num_layers+=1
                cur_C = C_out
        
        self.conv_layers = nn.Sequential(*conv_layers)
    
    def forward(self, x):
        '''
        x: (B, C, T)
        '''
        x = self.conv_layers(x)
        return x

class AudioPoseEncoder2D(nn.Module):
    '''
    (B, C, T) -> (B, 1, C, T) -> ... -> (B, C_out, T)
    '''
    def __init__(self):
        raise NotImplementedError('考虑到输出的T与输入相同，2D的形式不自然')

class AudioPoseEncoderRNN(nn.Module):
    '''
    RNN形式的encoder，主要改变维度
    (B, C, T)->(B, T, C)->(B, T, C_out)->(B, C_out, T)
    '''
    def __init__(self,
        C_in,
        hidden_size,
        num_layers,
        rnn_cell = 'gru',
        bidirectional=False
    ):
        super(AudioPoseEncoderRNN, self).__init__()
        if rnn_cell == 'gru':
            self.cell = nn.GRU(input_size = C_in, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional=bidirectional)
        elif rnn_cell == 'lstm':
            self.cell = nn.LSTM(input_size = C_in, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional=bidirectional)
        else:
            raise ValueError('invalid rnn cell:%s'%(rnn_cell))

    def forward(self, x, state=None):
        
        x = x.permute(0, 2, 1) 
        x, state = self.cell(x, state) 
        x = x.permute(0, 2, 1)
        
        return x 
        


class SeqEncoder2D(nn.Module):
    '''
    seq_encoder, 将一个seq数据encoding成一个vector
    (B, C, T)->(B, 1, C, T)->(B, 32, C/2, T/2)->...->(B, C_out)
    '''
    def __init__(self,
        C_in,
        T_in,
        C_out,
        min_layer_num=None
    ):
        super(SeqEncoder2D, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.T_in = T_in

        conv_layers = nn.ModuleList([])
        conv_layers.append(ConvNormRelu(
            in_channels=1,
            out_channels=32,
            type='2d'
        ))
        
        cur_C = 32
        cur_H = C_in
        cur_W = T_in
        num_layers = 1
        while (cur_C<C_out) or (cur_H>1) or (cur_W>1):
            ks = [3,3]
            st = [1,1]
    
            if cur_H>1:
                if cur_H>4:
                    ks[0] = 4
                    st[0] = 2
                else:
                    ks[0] = cur_H
                    st[0] = cur_H
            if cur_W>1:
                if cur_W>4:
                    ks[1] = 4
                    st[1] = 2
                else:
                    ks[1] = cur_W
                    st[1] = cur_W
            
            conv_layers.append(ConvNormRelu(
                in_channels=cur_C,
                out_channels=min(C_out, cur_C*2),
                type='2d',
                kernel_size=tuple(ks),
                stride=tuple(st)                
            ))
            cur_C = min(cur_C*2, C_out)
            if cur_H>1:
                if cur_H>4:
                    cur_H //= 2
                else: 
                    cur_H=1
            if cur_W>1:
                if cur_W>4:
                    cur_W //= 2
                else:
                    cur_W = 1
            num_layers += 1
    
        if min_layer_num is not None and (num_layers<min_layer_num):
            while num_layers < min_layer_num:
                conv_layers.append(ConvNormRelu(
                    in_channels=C_out,
                    out_channels=C_out,
                    type='2d',
                    kernel_size=1,
                    stride=1
                ))
                num_layers += 1
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.num_layers = num_layers

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return x.squeeze()

class SeqEncoder1D(nn.Module):
    '''
    使用1D卷积的seq encoder
    (B, C, T)->(B, D)
    仅仅encoding固定长度
    '''
    def __init__(self,
        C_in,
        C_out,
        T_in,
        min_layer_nums=None
    ):
        super(SeqEncoder1D, self).__init__()
        conv_layers=nn.ModuleList([])
        cur_C = C_in
        cur_T = T_in
        self.num_layers = 0
        while (cur_C<C_out) or (cur_T>1):
            ks=3
            st=1
            if cur_T>1:
                if cur_T>4:
                    ks = 4
                    st = 2
                else:
                    ks = cur_T
                    st = cur_T
            
            conv_layers.append(ConvNormRelu(
                in_channels=cur_C,
                out_channels=min(C_out, cur_C*2),
                type='1d',
                kernel_size=ks,
                stride=st
            ))
            cur_C = min(cur_C*2, C_out)
            if cur_T>1:
                if cur_T>4:
                    cur_T = cur_T // 2
                else:
                    cur_T = 1
            self.num_layers += 1

        if min_layer_nums is not None and (self.num_layers<min_layer_nums):
            while self.num_layers < min_layer_nums:
                conv_layers.append(ConvNormRelu(
                    in_channels=C_out,
                    out_channels=C_out,
                    type='1d',
                    kernel_size=1,
                    stride=1
                ))    
                self.num_layers += 1
        self.conv_layers = nn.Sequential(*conv_layers)
    
    def forward(self, x):
        x = self.conv_layers(x)
        return x.squeeze()

class SeqEncoderRNN(nn.Module):
    '''
    基于RNN的encoder，会简单一点
    (B, C, T) -> (B, T, C) -> (B, D)
    LSTM/GRU-FC
    '''
    def __init__(self,
        hidden_size,
        in_size,
        num_rnn_layers,
        rnn_cell='gru',
        bidirectional=False
    ):
        super(SeqEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.num_rnn_layers = num_rnn_layers
        self.bidirectional = bidirectional

        if rnn_cell == 'gru':
            self.cell = nn.GRU(input_size = self.in_size, hidden_size = self.hidden_size, num_layers = self.num_rnn_layers, batch_first = True, bidirectional=bidirectional)
        elif rnn_cell == 'lstm':
            self.cell = nn.LSTM(input_size = self.in_size, hidden_size = self.hidden_size, num_layers = self.num_rnn_layers, batch_first=True, bidirectional=bidirectional)
    
    def forward(self, x, state=None):
        
        x = x.permute(0, 2, 1)
        B,T,C = x.shape
        x,_ = self.cell(x, state)
        if self.bidirectional:
            out = torch.cat([x[:, -1, :self.hidden_size], x[:, 0, self.hidden_size:]], dim=-1)
        else:
            out = x[:, -1, :]
        assert out.shape[0] == B
        return out



class SeqDecoder2D(nn.Module):
    '''
    2D卷积形式的seq decoder，将一个feature decode成一个sequence，和SeqEncoder系列对应
    (B, D)->(B, D, 1, 1)->(B, C_out, C, T)->(B, C_out, T)
    因为输出只有一个维度，所以用2D的卷积不自然
    '''
    def __init__(self):
        super(SeqDecoder2D, self).__init__()
        raise NotImplementedError('2D 卷积的Seq decoder不自然')

class SeqDecoder1D(nn.Module):
    '''
    1D卷积形式的seq decoder，将一个feature vector decode成一个sequence，和SeqEncoder系列对应
    (B, D)->(B, D, 1)->...->(B, C_out, T)
    '''
    def __init__(self,
        D_in,
        C_out,
        T_out,
        min_layer_num=None
    ):
        super(SeqDecoder1D, self).__init__()
        self.T_out = T_out
        self.min_layer_num = min_layer_num

        cur_t = 1
        
        self.pre_conv = ConvNormRelu(
            in_channels=D_in,
            out_channels=C_out,
            type='1d'
        )
        self.num_layers = 1
        self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_layers = nn.ModuleList([])
        cur_t *= 2
        while cur_t<=T_out:
            self.conv_layers.append(ConvNormRelu(
                in_channels=C_out,
                out_channels=C_out,
                type='1d'
            ))
            cur_t *= 2
            self.num_layers += 1
        
        post_conv = nn.ModuleList([ConvNormRelu(
            in_channels=C_out,
            out_channels=C_out,
            type='1d'
        )])
        self.num_layers+=1
        if min_layer_num is not None and self.num_layers<min_layer_num:
            while self.num_layers<min_layer_num:
                post_conv.append(ConvNormRelu(
                    in_channels=C_out,
                    out_channels=C_out,
                    type='1d'
                ))
                self.num_layers += 1
        self.post_conv = nn.Sequential(*post_conv)

    def forward(self, x):
        
        x = x.unsqueeze(-1)
        x = self.pre_conv(x)
        for conv in self.conv_layers:
            x = self.upconv(x)
            x = conv(x)
        
        x = torch.nn.functional.interpolate(x, size=self.T_out, mode='nearest')
        x = self.post_conv(x)
        return x

class SeqDecoderRNN(nn.Module):
    '''
    使用RNN的seq的seq decoder
    (B, D)->(B, C_out, T)
    TODO:FC-LSTM-FC?,已知的是Audio2Body有严重的抖动
    TODO: 参考TriCon的做法
    '''
    def __init__(self,
        hidden_size,
        C_out,
        T_out,
        num_layers,
        rnn_cell = 'gru'
    ):
        super(SeqDecoderRNN, self).__init__()
        self.num_steps = T_out
        if rnn_cell == 'gru':
            self.cell = nn.GRU(input_size = C_out, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional=False)
        elif rnn_cell == 'lstm':
            self.cell = nn.LSTM(input_size = C_out, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional=False)
        else:
            raise ValueError('invalid rnn cell:%s'%(rnn_cell))
        
        self.fc = nn.Linear(hidden_size, C_out)

    def forward(self, hidden, frame_0):
        
        
        
        frame_0 = frame_0.permute(0, 2, 1)
        dec_input = frame_0
        outputs = []
        for i in range(self.num_steps):
            frame_out, hidden = self.cell(dec_input, hidden)
            frame_out = self.fc(frame_out)
            dec_input = frame_out
            outputs.append(frame_out)
        output = torch.cat(outputs, dim=1)
        return output.permute(0, 2, 1)




class SeqTranslator2D(nn.Module):
    '''
    (B, C, T)->(B, 1, C, T)-> ... -> (B, 1, C_out, T)
    '''
    def __init__(self):
        super(SeqTranslator2D, self).__init__()
        raise NotImplementedError('2D 卷积的Seq translator不自然')

class SeqTranslator1D(nn.Module):
    '''
    (B, C, T)->(B, C_out, T)
    '''
    def __init__(self,
        C_in,
        C_out,
        kernel_size=None,
        stride=None,
        min_layers_num=None
    ):
        super(SeqTranslator1D, self).__init__()
        
        conv_layers = nn.ModuleList([])
        conv_layers.append(ConvNormRelu(
            in_channels=C_in,
            out_channels=C_out,
            type='1d',
            kernel_size=kernel_size,
            stride=stride
        ))
        self.num_layers=1
        if min_layers_num is not None and self.num_layers<min_layers_num:
            while self.num_layers < min_layers_num:
                conv_layers.append(ConvNormRelu(
                    in_channels=C_out,
                    out_channels=C_out,
                    type='1d',
                    kernel_size=kernel_size,
                    stride=stride
                ))
                self.num_layers += 1
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv_layers(x)

class SeqTranslatorRNN(nn.Module):
    '''
    使用RNN的seq decoder
    (B, C, T)->(B, C_out, T)
    LSTM-FC
    '''
    def __init__(self,
        C_in,
        C_out,
        hidden_size,
        num_layers,
        rnn_cell = 'gru'
    ):
        super(SeqTranslatorRNN, self).__init__()
        
        if rnn_cell == 'gru':
            self.enc_cell = nn.GRU(input_size = C_in, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional=False)
            self.dec_cell = nn.GRU(input_size = C_out, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional=False)
        elif rnn_cell == 'lstm':
            self.enc_cell = nn.LSTM(input_size = C_in, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional=False)
            self.dec_cell = nn.LSTM(input_size = C_out, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional=False)
        else:
            raise ValueError('invalid rnn cell:%s'%(rnn_cell))
        
        self.fc = nn.Linear(hidden_size, C_out)

    def forward(self, x, frame_0):
        
        num_steps = x.shape[-1]
        x = x.permute(0, 2, 1)
        frame_0 = frame_0.permute(0, 2, 1)
        _, hidden = self.enc_cell(x, None)

        outputs = []
        for i in range(num_steps):
            inputs = frame_0
            output_frame, hidden = self.dec_cell(inputs, hidden)
            output_frame = self.fc(output_frame)
            frame_0 = output_frame
            outputs.append(output_frame)
        outputs = torch.cat(outputs, dim=1)
        return outputs.permute(0, 2, 1)

class ResBlock(nn.Module):
    def __init__(self, 
        input_dim, 
        fc_dim, 
        afn, 
        nfn
    ):
        '''
        afn: activation fn
        nfn: normalization fn
        '''
        super(ResBlock, self).__init__()

        self.input_dim=input_dim
        self.fc_dim=fc_dim
        self.afn=afn
        self.nfn=nfn

        if self.afn !='relu':
            raise ValueError('Wrong')
        
        if self.nfn=='layer_norm':
            raise ValueError('wrong')

        self.layers=nn.Sequential(
            nn.Linear(self.input_dim, self.fc_dim // 2),
            nn.ReLU(),
            nn.Linear(self.fc_dim//2, self.fc_dim//2),
            nn.ReLU(),
            nn.Linear(self.fc_dim//2, self.fc_dim),
            nn.ReLU()
        )
        
        self.shortcut_layer=nn.Sequential(
            nn.Linear(self.input_dim, self.fc_dim),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.layers(inputs)+self.shortcut_layer(inputs)

if __name__ == '__main__':
    
    
    import numpy as np
    random_C = np.random.randint(100, 1000)
    print(random_C)
    test_model = SeqTranslatorRNN(
        C_in=32, 
        C_out=108,
        hidden_size=64,
        num_layers=1,
        rnn_cell='gru'
    )
    dummy_input = torch.randn([10, 32, 25])
    dummy_frame_0 = torch.randn([10, 108, 1])
    output = test_model(dummy_input, dummy_frame_0)
    print(output.shape)