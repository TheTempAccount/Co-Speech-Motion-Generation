import numpy as np
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_layers=1, one_hot=False, num_keypoints=25+21+21):
        super(Discriminator, self).__init__()
        self.human_size=num_keypoints*2
        self.one_hot = one_hot
        self.input_size=self.human_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.gru=nn.GRU(self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.dense1=nn.Linear(self.hidden_size, 1)

    def forward(self, seq, state=None):#action, state
        p,hidden=self.gru(seq, state)
        p = p[-1,...].squeeze(0)#(batch_size, hidden_size)
        prob=torch.sigmoid(self.dense1(p))#(batch_size, 1)
        return prob
    
    def init_hidden(self):
        raise NotImplementedError

