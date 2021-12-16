import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from repro_nets.freeMo_paper_code.layer_utils import ResNet_Block

class LatentDecoder(nn.Module):
    def __init__(self, 
        embed_size, 
        noise_size, 
        dec_hidden_size, 
        content_dim,
        dec_interaction,
        T_layer_norm,
        num_dec_fc_layers,
        latent_fc_layers,
        device
    ):
        super(LatentDecoder, self).__init__()
        self.embed_size=embed_size
        self.noise_size=noise_size
        self.style_size=self.embed_size-content_dim
        self.dec_hidden_size=dec_hidden_size
        self.device = device
        self.latent_fc_layers = latent_fc_layers
        self.T_layer_norm = T_layer_norm

        #首先将noise映射到style_size上, 原文中并未使用activation
        self.map_noise=nn.Sequential(
            nn.Linear(self.noise_size, self.style_size),
            nn.LayerNorm(self.style_size)
        )

        self.interaction_fc=self.get_interaction_fc(dec_interaction)

        if T_layer_norm:
            self.style_norm_prev=nn.LayerNorm(self.style_size)
            self.style_norm_post=nn.LayerNorm(self.style_size)

        #下面是得到了新的style_code之后，将其与content code concat之后的处理
        dec_fc_layers=[]
        for layer_i in range(num_dec_fc_layers):
            dec_fc_layers.append(
                nn.Sequential(
                    ResNet_Block(self.embed_size, self.embed_size, afn='relu', nfn=None),
                    nn.LayerNorm(self.embed_size)
                )
            )
        
        self.dec_fc_layers=nn.Sequential(*dec_fc_layers)

        #最后得到seq_decoder的hidden_state, 因为是LSTM，所以有一个h和一个c
        self.h_fc=nn.Linear(self.embed_size, self.dec_hidden_size)
        self.h_agn=nn.Tanh()#这里有一个tanh激活
        self.c_hc=nn.Linear(self.embed_size, self.dec_hidden_size)#而这里没有

    def concat_fn(self):
        
        return nn.Sequential(
            nn.Linear(self.embed_size, self.style_size),
            nn.ReLU(),
            nn.LayerNorm(self.style_size),
            nn.Linear(self.style_size, self.style_size)
        )
    
    def add_fc(self):
        middle_layers=[]
        middle_layers.append(
            nn.Sequential(
            ResNet_Block(self.embed_size, self.style_size, afn='relu', nfn=None),
            nn.LayerNorm(self.style_size))
        )

        for layer_i in range(self.latent_fc_layers-1):
            middle_layers.append(
                nn.Sequential(
                ResNet_Block(self.style_size, self.style_size, afn='relu', nfn=None),
                nn.LayerNorm(self.style_size))
            )
        middle_layers.append(nn.Linear(self.style_size, self.style_size))
        return nn.Sequential(*middle_layers)

    def get_interaction_fc(self, mode):
        if mode=='add':
            return self.add_fc()
        elif mode=='concat':
            return self.concat_fn()

    def forward(self, latent_input, prev_content, prev_style, data_type='trans'):
        '''
        latent_input应该就是noise
        '''
        if data_type == 'trans':
            noise=self.map_noise(latent_input)
            #可以考虑在这里将noise设置为zero_vector
            concat=torch.cat([noise, prev_style], dim=1)
            assert concat.shape[-1]==self.style_size*2

            T_new=self.interaction_fc(concat)

            if self.T_layer_norm>0:
                T_new=self.style_norm_prev(T_new)
            
            new_style=prev_style+T_new
            if self.T_layer_norm>0:
                new_style=self.style_norm_post(new_style)
            
            # result={}
            # result['new_style']=new_style.clone()

            new_code=torch.cat([prev_content, new_style], dim=1)
            new_code=self.dec_fc_layers(new_code)

            h0=self.h_fc(new_code)
            h0=self.h_agn(h0)#(batch_size, hidden_size)

            c0=self.c_hc(new_code)

            h0=h0.unsqueeze(0)#加上lstm_layer维度
            c0=c0.unsqueeze(0)

            # result['dec_embedding']=(h0, c0)
            return (h0, c0), new_style
        elif data_type=='recons':
            #data_type == zero, 直接使用input进行prediction
            result={}
            result['new_style']=prev_style.clone()
            new_code=torch.cat([prev_content, prev_style], dim=1)#实际上prev_style已经在之前norm过了
            new_code=self.dec_fc_layers(new_code)

            h0=self.h_fc(new_code)
            h0=self.h_agn(h0)#(batch_size, hidden_size)

            c0=self.c_hc(new_code)

            h0=h0.unsqueeze(0)#加上lstm_layer维度
            c0=c0.unsqueeze(0)

            return (h0, c0), prev_style
        else:
            raise ValueError

if __name__ == '__main__':
    test_model = LatentDecoder(
        embed_size=1024,
        noise_size=512,
        dec_hidden_size=1024,
        content_dim=512,
        dec_interaction='add',
        T_layer_norm=True,
        num_dec_fc_layers=1,
        latent_fc_layers=1,
        device=torch.device('cpu')   
    )

    latent_input = torch.randn(64, 512)
    prev_content = torch.randn(64, 512)
    prev_style = torch.randn(64, 512)
    output = test_model(
        latent_input,
        prev_content,
        prev_style,
        data_type = 'trans',
    )

    print(output[0].shape)
    print(output[1][0].shape, output[1][1].shape)