import torch
import torch.nn as nn

class SeqDecoder(nn.Module):
    def __init__(self, num_keypoints, embed_size, args, num_steps, hidden_size, device):
        super(SeqDecoder, self).__init__()

        self.num_keypoints=num_keypoints
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.args=args
        self.num_steps=num_steps
        self.device=device
        self.lstm=nn.LSTM(input_size=self.embed_size // 2, hidden_size=self.hidden_size, num_layers=1)
        self.fc=nn.Linear(self.hidden_size, self.num_keypoints*2)

    def forward(self, prev_states, dec_input):
        assert dec_input.shape[-1]==self.embed_size // 2
        assert len(prev_states)==2
        dec_input=dec_input.unsqueeze(0)
        outputs=[]
        for _ in range(self.num_steps):
            output, prev_states=self.lstm(dec_input, prev_states)
            outputs.append(output.squeeze(0))
        outputs=torch.stack(outputs, dim=0)
        outputs=self.fc(outputs)
        results={}
        results['keypoints_output']=outputs
        results['state']=prev_states
        return results