import torch
import torch.nn as nn
import torch.nn.functional as F
from .latent_decoder import LatentDecoder
from .latent_encoder import LatentEncoder
from .seq_decoder import SeqDecoder
from .seq_encoder import SeqEncoder
from .audio2pose import Audio2Pose

class Core(nn.Module):
    def __init__(self, args, device):
        super(Core, self).__init__()
        self.args=args

        self.device = device
        self.seq_encoder=SeqEncoder(self.args.num_keypoints, self.args.embed_size, self.args, self.args.hidden_size, device=self.device)
        self.seq_decoder=SeqDecoder(self.args.num_keypoints, self.args.embed_size, self.args, self.args.seq_length_out, self.args.hidden_size, device=self.device)
        self.latent_encoder=LatentEncoder(self.args.embed_size, self.args.noise_size, self.args, self.device)
        self.latent_decoder=LatentDecoder(self.args.embed_size, self.args.noise_size, self.args.hidden_size, self.args, device=self.device)
        if self.args.audio_decoding:
            self.aud2pose = Audio2Pose(self.args.num_keypoints, self.args.embed_size, self.args.augmentation)

    def sample(self, inputs, data_type='trans'):
        outputs=self.seq_encoding(inputs)
        his_style=outputs['his_style']
        his_content=outputs['his_content']
        if data_type=='trans':
            noise=torch.randn_like(his_style, device=self.device)
        else:
            noise=torch.zeros_like(his_style, device=self.device)
        outputs['latent']=noise
        outputs=self.latent_decoding(inputs, outputs)
        outputs=self.seq_decoding(inputs, outputs)

        if data_type == 'zero':
            if self.args.audio_decoding:
                outputs = self.audio_decoding(inputs, outputs)
                
        return outputs

    def seq_encoding(self, input):
        results={}
        his_landmarks=input['his_landmarks']
        his_encoding=self.seq_encoder(his_landmarks)
        results['his_features']=his_encoding['features']
        if self.args.content_dim>0:
            results['his_content']=his_encoding['content']
            results['his_style']=his_encoding['style']
        
        # if self.training:
        if 'fut_landmarks' in list(input.keys()):
            fut_landmarks=input['fut_landmarks']
            fut_encoding=self.seq_encoder(fut_landmarks)
            results['fut_features']=fut_encoding['features']
            if self.args.content_dim>0:
                results['fut_content']=fut_encoding['content']
                results['fut_style']=fut_encoding['style']
            
        return results

    def latent_encoding(self, inputs, outputs):
        
        encoding=self.latent_encoder(outputs['his_style'],outputs['fut_style'])
        outputs['latent']=encoding['latent']
        outputs['mu_latent']=encoding['mu']
        outputs['logvar_latent']=encoding['var']
        return outputs

    def latent_decoding(self, inputs, outputs, data_type='trans'):
        
        decoding=self.latent_decoder(outputs['latent'], outputs['his_content'], outputs['his_style'], data_type=data_type)
        outputs['dec_embedding']=decoding['dec_embedding']
        outputs['dec_style']=decoding['new_style']
        return outputs

    def seq_decoding(self, inputs, outputs):
        prev_state=outputs['dec_embedding']
        dec_style=outputs['dec_style']

        preds=self.seq_decoder(prev_state, dec_style)
        outputs['fut_landmarks']=preds['keypoints_output']
        return outputs
    
    def audio_decoding(self, inputs, outputs):
        audio_feat = inputs['mfcc_feature']
        dec_style=outputs['dec_style']

        kp_residual = self.aud2pose(audio_feat, dec_style)
        outputs['residual'] = kp_residual
        outputs['fut_landmarks'] = outputs['fut_landmarks'] + outputs['residual']
        return outputs