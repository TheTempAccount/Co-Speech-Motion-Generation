import torch
import torch.nn as nn

class SeqEncoder(nn.Module):
    def __init__(self, num_keypoints, embed_size, args, hidden_size, device):
        super(SeqEncoder, self).__init__()

        self.num_keypoints=num_keypoints
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.args=args
        self.device=device
        self.lstm=nn.LSTM(input_size=self.num_keypoints*2, hidden_size=self.hidden_size, num_layers=1)
        self.fc=nn.Linear(self.hidden_size, self.embed_size)

        if self.args.content_dim > 0:
            if self.args.T_layer_norm>0 :
                self.layer_norm=nn.LayerNorm(self.embed_size//2)


    def forward(self, inputs):
        assert inputs.shape[-1]==self.num_keypoints*2
        S, B = inputs.shape[0], inputs.shape[1]
        outputs, state=self.lstm(inputs)
        outputs=outputs[-1,:,:]

        assert outputs.shape[0] == B
        outputs=self.fc(outputs)
        
        embedding={}
        embedding['features']=outputs
        if self.args.content_dim:
            if self.args.T_layer_norm>0 :
                embedding['content']=outputs[:,0:self.args.content_dim]
                embedding['style']=self.layer_norm(outputs[:,self.args.content_dim:])
            
        return embedding