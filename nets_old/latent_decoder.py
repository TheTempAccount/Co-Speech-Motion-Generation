import torch
import torch.nn as nn
from .layer_utils import ResNet_Block

class LatentDecoder(nn.Module):
    def __init__(self, embed_size, noise_size, dec_hidden_size, args, device):
        super(LatentDecoder, self).__init__()
        self.embed_size=embed_size
        self.noise_size=noise_size
        self.args=args
        self.style_size=self.embed_size-self.args.content_dim
        self.dec_hidden_size=dec_hidden_size
        self.device = device

        self.map_noise=nn.Sequential(
            nn.Linear(self.noise_size, self.style_size),
            nn.LayerNorm(self.style_size)
        )

        self.interaction_fc=self.get_interaction_fc(self.args.dec_interaction)

        if self.args.T_layer_norm>0:
            self.style_norm_prev=nn.LayerNorm(self.style_size)
            self.style_norm_post=nn.LayerNorm(self.style_size)

        dec_fc_layers=[]
        for layer_i in range(self.args.dec_fc_layers):
            dec_fc_layers.append(
                nn.Sequential(
                    ResNet_Block(self.embed_size, self.embed_size, afn='relu', nfn=None),
                    nn.LayerNorm(self.embed_size)
                )
            )
        
        self.dec_fc_layers=nn.Sequential(*dec_fc_layers)

        self.h_fc=nn.Linear(self.embed_size, self.dec_hidden_size)
        self.h_agn=nn.Tanh()
        self.c_hc=nn.Linear(self.embed_size, self.dec_hidden_size)

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

        for layer_i in range(self.args.latent_fc_layers-1):
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
        if data_type == 'trans':
            noise=self.map_noise(latent_input)
            concat=torch.cat([noise, prev_style], dim=1)
            assert concat.shape[-1]==self.style_size*2

            T_new=self.interaction_fc(concat)

            if self.args.T_layer_norm>0:
                T_new=self.style_norm_prev(T_new)
            
            new_style=prev_style+T_new
            if self.args.T_layer_norm>0:
                new_style=self.style_norm_post(new_style)
            
            result={}
            result['new_style']=new_style.clone()

            new_code=torch.cat([prev_content, new_style], dim=1)
            new_code=self.dec_fc_layers(new_code)

            h0=self.h_fc(new_code)
            h0=self.h_agn(h0)

            c0=self.c_hc(new_code)

            h0=h0.unsqueeze(0)
            c0=c0.unsqueeze(0)

            result['dec_embedding']=(h0, c0)
        elif data_type=='recons':
            result={}
            result['new_style']=prev_style.clone()
            new_code=torch.cat([prev_content, prev_style], dim=1)
            new_code=self.dec_fc_layers(new_code)

            h0=self.h_fc(new_code)
            h0=self.h_agn(h0)

            c0=self.c_hc(new_code)

            h0=h0.unsqueeze(0)
            c0=c0.unsqueeze(0)

            result['dec_embedding']=(h0, c0)
        else:
            raise ValueError

        return result
