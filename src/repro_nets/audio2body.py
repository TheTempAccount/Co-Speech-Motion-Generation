import os
import sys
sys.path.append(os.getcwd())

from nets.layers import *
import torch
import torch.nn as nn
import torch.nn.init as init

class Generator(nn.Module):

    def __init__(self, 
        in_dim,
        hidden_dim,
        dropout_p,
        output_dim,
        rnn_cell='lstm',
        num_layers=1
    ):
        super(Generator, self).__init__()

        

        
        self.cell = AudioPoseEncoderRNN(
            C_in=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            rnn_cell=rnn_cell
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.initialize()

    def initialize(self):
        
        for layer in self.cell.cell._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.cell.cell, param_name)
                    init.xavier_normal_(weight.data)
                else:
                    bias = getattr(self.cell.cell, param_name)
                    init.uniform_(bias.data, 0.25, 0.5)

        
        init.xavier_normal_(self.fc.weight.data)
        init.constant_(self.fc.bias.data, 0)

    def forward(self, x):
        
        
        x = self.cell(x)
        x = x.permute(0, 2, 1)
        
        x = self.dropout(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x

if __name__ == "__main__":
    test_model = Generator(
        in_dim=64,
        hidden_dim=256,
        dropout_p=0.2,
        output_dim=108,
    )
    dummy_input = torch.randn([64, 64, 120])
    output = test_model(dummy_input)
    print(output.shape)