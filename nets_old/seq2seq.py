import numpy as np
import torch
import torch.nn as nn

def nll_gauss(mean, std, x):
    pi = torch.FloatTensor([np.pi])
    if mean.is_cuda:
        pi=pi.cuda()
    nll_element=(x-mean).pow(2) / std.pow(2) + 2*torch.log(std) + torch.log(2*pi)
    return 0.5 * torch.sum(nll_element)

def reparam_sample_gauss(mean, std):
    eps=torch.FloatTensor(std.size()).normal_()
    if mean.is_cuda:
        eps=eps.cuda()
    return eps.mul(std).add_(mean)

def reverse_sample_gauss(std, sample):
    raise NotImplementedError

class DecoderWrapper(nn.Module):
    def __init__(self, model, hidden_size, output_size, residual, one_hot, device):
        super(DecoderWrapper, self).__init__()
        self.model=model
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.residual=residual
        self.linear=nn.Linear(hidden_size, output_size)
        self.device=device
        self.one_hot=one_hot
        #init
        torch.nn.init.uniform_(self.linear.weight, -0.04, 0.04)
    
    def forward(self, inputs, state, target_seq_len):
        time_seq_len, batch_size, input_size=inputs.shape
        if self.one_hot:
            assert self.output_size==input_size
        else:
            assert self.output_size==input_size

        outputs=torch.zeros((target_seq_len, batch_size, input_size)).to(self.device)
        for i in range(target_seq_len):
            temp, state=self.model(inputs, state)
            new_frame=inputs.clone()
            new_frame[:,:,:self.output_size]=self.linear(temp) + inputs[:,:,:self.output_size] if self.residual else self.linear(temp)
            outputs[i]=new_frame
            inputs=new_frame
        
        return outputs, state

class StochasticDecoderWrapper(nn.Module):
    def __init__(self, model, hidden_size, output_size, residual, one_hot, device):
        super(StochasticDecoderWrapper, self).__init__()
        self.model=model
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.residual=residual
        self.device=device
        self.one_hot=one_hot

        self.mean_estimator=nn.Linear(self.hidden_size,self.output_size)
        self.std_estimator=nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.Softplus()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05,0.05)

    def forward(self, inputs, state, target_seq_len):
        seq_len, batch_size, input_size=inputs.shape
        output_mean=torch.zeros(target_seq_len, batch_size, self.output_size).to(self.device)
        output_std=torch.zeros(target_seq_len, batch_size, self.output_size).to(self.device)
        output_frame=torch.zeros(target_seq_len, batch_size, input_size).to(self.device)
        if self.one_hot:
            assert input_size==self.output_size
        else:
            assert input_size==self.output_size

        for i in range(target_seq_len):
            temp,state=self.model(inputs, state)
            mean=self.mean_estimator(temp)
            std=self.std_estimator(temp)
            next_frame=inputs.clone()
            next_frame[:,:,:self.output_size]=reparam_sample_gauss(mean, std) + inputs[:,:,:self.output_size] if self.residual else reparam_sample_gauss(mean, std)
            output_frame[i]=next_frame
            output_mean[i]=mean
            output_std[i]=std
            inputs=next_frame
        return output_mean, output_std, output_frame, state


class Seq2Seq(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, residual, stochastic, output_size, one_hot, device, input_size):
        super(Seq2Seq, self).__init__()
        self.encoder_hidden_size=encoder_hidden_size
        self.decoder_hidden_size=decoder_hidden_size
        self.one_hot=one_hot
        self.input_size=input_size
        self.output_size=output_size
        self.residual=residual
        self.stochastic=stochastic
        self.device=device

        self.encoder=nn.GRU(input_size=self.input_size, hidden_size=self.encoder_hidden_size,num_layers=1, batch_first=False)
        self.decoder=nn.GRU(input_size=self.output_size, hidden_size=self.decoder_hidden_size, num_layers=1, batch_first=False)
        if not self.stochastic:
            self.decoder=DecoderWrapper(self.decoder, self.decoder_hidden_size, self.output_size, self.residual, self.one_hot, device)#wrapper
            self.loss=nn.MSELoss(reduction='sum')
        else:
            print('using nll_gauss')
            self.decoder=StochasticDecoderWrapper(self.decoder, self.decoder_hidden_size, self.output_size, self.residual, self.one_hot, device)
            self.loss=nll_gauss
        
    def forward(self, encoder_input, decoder_input, target_seq_len):
        src_seq_len, batch_size, src_input_size=encoder_input.shape
        dst_seq_len, _, dst_input_size=decoder_input.shape

        features, state=self.encoder(encoder_input, None)
        last_frame=decoder_input[0,:,:].view(1,-1,self.output_size)
        if not self.stochastic:
            output, state=self.decoder(last_frame, state, target_seq_len)
            return output, state
        else:
            mean, std, output, state=self.decoder(last_frame, state, target_seq_len)
            return mean, std, output, state

    def encode(self):
        '''
        '''
        pass
        
    
class Seq2Seq_with_noise(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, residual, stochastic, output_size, one_hot, device, mean=None, std=None):
        super(Seq2Seq, self).__init__()
        self.encoder_hidden_size=encoder_hidden_size
        self.decoder_hidden_size=decoder_hidden_size
        self.one_hot=one_hot
        self.input_size=25*2 if one_hot else 25*2
        self.output_size=25*2
        self.residual=residual
        self.stochastic=stochastic
        self.output_size=output_size
        self.device=device
        self.mean=mean
        self.std=std

        self.encoder=nn.GRU(input_size=self.input_size, hidden_size=self.encoder_hidden_size,num_layers=1, batch_first=False)
        self.decoder=nn.GRU(input_size=self.input_size, hidden_size=self.decoder_hidden_size, num_layers=1, batch_first=False)
        if not self.stochastic:
            self.decoder=DecoderWrapper(self.decoder, self.decoder_hidden_size, self.output_size, self.residual, self.one_hot, device)#wrapper
            self.loss=nn.MSELoss(reduction='sum')
        else:
            print('using nll_gauss')
            self.decoder=StochasticDecoderWrapper(self.decoder, self.decoder_hidden_size, self.output_size, self.residual, self.one_hot, device)
            self.loss=nll_gauss
        
    def forward(self, encoder_input, decoder_input, source_seq_len, target_seq_len):
        src_seq_len, batch_size, src_input_size=encoder_input.shape
        dst_seq_len, batch_size, dst_input_size=decoder_input.shape
        assert src_input_size==dst_input_size==self.input_size

        features, state=self.encoder(encoder_input, None)

        if self.mean is not None:
            pass


        last_frame=decoder_input[0,:,:].view(1,-1,self.input_size)
        if not self.stochastic:
            output, state=self.decoder(last_frame, state, target_seq_len)
            return output, state
        else:
            mean, std, output, state=self.decoder(last_frame, state, target_seq_len)
            return mean, std, output, state

class Seq2Seq_VAE(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, residual, stochastic, output_size, one_hot, kld_weight, device, num_keypoints=25):
        super(Seq2Seq_VAE, self).__init__()
        self.encoder_hidden_size=encoder_hidden_size
        self.decoder_hidden_size=decoder_hidden_size
        self.one_hot=one_hot
        self.num_keypoints=num_keypoints
        self.input_size=self.num_keypoints*2 if one_hot else self.num_keypoints*2
        self.output_size=self.num_keypoints*2
        self.residual=residual
        self.stochastic=stochastic
        self.output_size=output_size
        self.device=device
        self.kld_weight=kld_weight

        self.encoder=nn.GRU(input_size=self.input_size, hidden_size=self.encoder_hidden_size,num_layers=1, batch_first=False)
        self.decoder=nn.GRU(input_size=self.input_size, hidden_size=self.decoder_hidden_size, num_layers=1, batch_first=False)
        if not self.stochastic:
            self.decoder=DecoderWrapper(self.decoder, self.decoder_hidden_size, self.output_size, self.residual, self.one_hot, device)#wrapper
            self.loss=nn.MSELoss(reduction='sum')
        else:
            print('using nll_gauss')
            self.decoder=StochasticDecoderWrapper(self.decoder, self.decoder_hidden_size, self.output_size, self.residual, self.one_hot, device)
            self.loss=nll_gauss

        self.fc_mu=nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.fc_var=nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
    
    def encode(self, input):
        features, state=self.encoder(input, None)

        state=state.squeeze(0)
        assert state.shape[-1]==self.encoder_hidden_size
        mu=self.fc_mu(state)
        log_var=self.fc_var(state)

        return [mu, log_var]

    def decode(self, last_frame, state, target_seq_len):
        if not self.stochastic:
            output, state=self.decoder(last_frame, state, target_seq_len)
            return output, state
        else:
            mean, std, output, state=self.decoder(last_frame, state, target_seq_len)
            return mean, std, output, state

    def reparameterize(self, mu, logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std, device=self.device)
        z=eps*std+mu
        z=z.unsqueeze(0)
        return z

    def forward(self, encoder_input, decoder_input, source_seq_len, target_seq_len):
        src_seq_len, batch_size, src_input_size=encoder_input.shape
        dst_seq_len, batch_size, dst_input_size=decoder_input.shape
        assert src_input_size==dst_input_size==self.input_size

        mu, log_var = self.encode(encoder_input)
        z=self.reparameterize(mu, log_var)
    
        last_frame=decoder_input[0,:,:].view(1,-1,self.input_size)
        if not self.stochastic:
            output, state=self.decode(last_frame, z, target_seq_len)
            return [output, state, mu, log_var]
        else:
            mean, std, output, state=self.decode(last_frame, z, target_seq_len)
            return [mean, std, output, state, mu, log_var]

    def loss_function(self, pred, target, mu, log_var):
        recons_loss=self.loss(pred, target)
        kld_weight=self.kld_weight

        kld_loss = torch.mean(-0.5*torch.sum(1+log_var-mu**2-log_var.exp(), dim=1), dim=0)
        loss=recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'reconstruction_loss': recons_loss.detach().cpu().item(), 'KLD': kld_loss.detach().cpu().item()}

class Seq2Seq_CVAE(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, residual, stochastic, output_size, one_hot, device, num_keypoints):
        super(Seq2Seq, self).__init__()
        self.encoder_hidden_size=encoder_hidden_size
        self.decoder_hidden_size=decoder_hidden_size
        self.one_hot=one_hot
        self.num_keypoints=num_keypoints
        self.input_size=self.num_keypoints*2 if one_hot else self.num_keypoints*2
        self.output_size=25*2
        self.residual=residual
        self.stochastic=stochastic
        self.output_size=output_size
        self.device=device

        self.encoder=nn.GRU(input_size=self.input_size, hidden_size=self.encoder_hidden_size,num_layers=1, batch_first=False)
        self.decoder=nn.GRU(input_size=self.input_size, hidden_size=self.decoder_hidden_size, num_layers=1, batch_first=False)
        if not self.stochastic:
            self.decoder=DecoderWrapper(self.decoder, self.decoder_hidden_size, self.output_size, self.residual, self.one_hot, device)#wrapper
            self.loss=nn.MSELoss(reduction='sum')
        else:
            print('using nll_gauss')
            self.decoder=StochasticDecoderWrapper(self.decoder, self.decoder_hidden_size, self.output_size, self.residual, self.one_hot, device)
            self.loss=nll_gauss
        
    def forward(self, encoder_input, decoder_input, source_seq_len, target_seq_len):
        src_seq_len, batch_size, src_input_size=encoder_input.shape
        dst_seq_len, batch_size, dst_input_size=decoder_input.shape
        assert src_input_size==dst_input_size==self.input_size

        features, state=self.encoder(encoder_input, None)
        last_frame=decoder_input[0,:,:].view(1,-1,self.input_size)
        if not self.stochastic:
            output, state=self.decoder(last_frame, state, target_seq_len)
            return output, state
        else:
            mean, std, output, state=self.decoder(last_frame, state, target_seq_len)
            return mean, std, output, state