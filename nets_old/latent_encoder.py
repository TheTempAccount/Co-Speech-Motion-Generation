import torch
from .layer_utils import ResNet_Block
import torch.nn as nn

class LatentEncoder(nn.Module):
    def __init__(self, embed_size, noise_size, args, device):
        super(LatentEncoder, self).__init__()
        self.embed_size=embed_size
        self.noise_size=noise_size
        self.args=args
        self.device=device
        self.style_size=self.embed_size-self.args.content_dim

        self.layer_norm_1=nn.LayerNorm(self.style_size)

        latent_fc_layers=[]
        latent_fc_layers.append(
            nn.Sequential(
                ResNet_Block(self.style_size, self.embed_size, afn='relu', nfn=None),
                nn.LayerNorm(self.embed_size)
            )
        )
        for layer_i in range(self.args.latent_fc_layers-1):
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
        outputs=self.layer_norm_1(fut_style-his_style)
        outputs=self.latent_fc_layers(outputs)
        mu=self.mu_fc(outputs)
        log_var=self.var_fc(outputs)
        sample_code=self.sample(mu, log_var)

        results={}
        results['mu']=mu
        results['var']=log_var
        results['latent']=sample_code
        return results

    def sample(self, mu, log_var):
        std=torch.exp(0.5*log_var)
        eps=torch.randn_like(std, device=self.device)
        z=eps*std+mu
        return z

        