import torch
import torch.nn as nn

class ResNet_Block(nn.Module):
    def __init__(self, input_dim, fc_dim, afn, nfn):
        '''
        afn: activation fn
        nfn: normalization fn
        '''
        super(ResNet_Block, self).__init__()

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
