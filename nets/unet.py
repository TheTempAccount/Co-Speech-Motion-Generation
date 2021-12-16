import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.conv import Conv1d

class SCSEAttention(nn.Module):
    def __init__(self,
        C_in,
        reduction=16,
        type='1d'
    ):
        super().__init__()
        #channel wise, 1x1->conv relu conv -> sigmoid
        if type == '1d':
            self.cSE = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(C_in, C_in//reduction, 1, 1),
                nn.ReLU(),
                nn.Conv1d(C_in // reduction, C_in, 1, 1),
                nn.Sigmoid(),
            )
            #spatial wise
            self.sSE = nn.Sequential(
                nn.Conv1d(C_in, 1, 1, 1),
                nn.Sigmoid()
            )
        elif type == '2d':
            self.cSE = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(C_in, C_in//reduction, 1, 1),
                nn.ReLU(),
                nn.Conv2d(C_in // reduction, C_in, 1, 1),
                nn.Sigmoid(),
            )
            #spatial wise
            self.sSE = nn.Sequential(
                nn.Conv2d(C_in, 1, 1, 1),
                nn.Sigmoid()
            )
        else: raise ValueError(type)

    def forward(self, x):
        return x*self.cSE(x) + x*self.sSE(x)

class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEAttention(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class UnetDecoder(nn.Module):
    def __init__(self):
        '''
        原始的Unet，对x进行上采样，然后concat，然后经过attention，conv， conv， attention
        '''
        super().__init__()

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

    