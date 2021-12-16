import os
import sys
sys.path.append(os.getcwd())

from logging import log
import torch
from torch._C import device
from repro_nets.freeMo_paper_code.layer_utils import ResNet_Block
import torch.nn as nn

class LatentEncoder(nn.Module):
    def __init__(self, 
        embed_size, 
        noise_size, 
        content_dim, 
        num_latent_fc_layers,
        device
    ):
        super(LatentEncoder, self).__init__()
        self.embed_size=embed_size
        self.noise_size=noise_size
        self.device=device
        #TODO: 原文这里虽然是style作为input，但是fc还是上升到了embed_size
        self.style_size=self.embed_size-content_dim

        self.layer_norm_1=nn.LayerNorm(self.style_size)#首先对style之差进行layer norm

        latent_fc_layers=[]
        latent_fc_layers.append(
            nn.Sequential(
                ResNet_Block(self.style_size, self.embed_size, afn='relu', nfn=None),
                nn.LayerNorm(self.embed_size)
            )
        )
        for layer_i in range(num_latent_fc_layers-1):
            latent_fc_layers.append(
                nn.Sequential(
                    ResNet_Block(self.embed_size, self.embed_size, afn='relu', nfn=None),
                    nn.LayerNorm(self.embed_size)
                )
            )

        self.latent_fc_layers=nn.Sequential(*latent_fc_layers)

        self.mu_fc=nn.Linear(self.embed_size, self.noise_size)
        self.var_fc=nn.Linear(self.embed_size, self.noise_size)
    
    def forward(self, his_style, fut_style):
        assert his_style.shape[-1]==self.style_size
        outputs=self.layer_norm_1(fut_style-his_style)#(batch_size, style_size)
        outputs=self.latent_fc_layers(outputs)
        mu=self.mu_fc(outputs)
        log_var=self.var_fc(outputs)
        sample_code=self.sample(mu, log_var)

        return mu, log_var, sample_code

    def sample(self, mu, log_var):
        std=torch.exp(0.5*log_var)
        eps=torch.randn_like(std, device=self.device)
        z=eps*std+mu
        return z

if __name__ == "__main__":
    test_model = LatentEncoder(
        embed_size=1024,
        noise_size=512,
        content_dim=512,
        num_latent_fc_layers=1,
        device=torch.device('cpu')
    )        
    his_style = torch.randn([64, 512])
    fut_style = torch.randn([64, 512])

    mu, log_var, z = test_model(his_style, fut_style)
    print(mu.shape, log_var.shape, z.shape)
