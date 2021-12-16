import torch
from torch._C import device
import torch.nn as nn

class SeqDecoder(nn.Module):
    def __init__(self, 
        pose_dim, 
        embed_size, 
        num_steps, 
        hidden_size, 
        device
    ):
        '''
        LSTM - FC
        从embed_size到num_keypoints的映射
        '''
        super(SeqDecoder, self).__init__()

        self.pose_dim = pose_dim
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.num_steps=num_steps
        self.device=device
        self.lstm=nn.LSTM(input_size=self.embed_size // 2, hidden_size=self.hidden_size, num_layers=1)
        self.fc=nn.Linear(self.hidden_size, pose_dim)

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
        
        
        
        
        return outputs, prev_states

if __name__ == "__main__":
    test_model = SeqDecoder(
        108,
        1024,
        25,
        1024,
        device=torch.device('cpu')
    )
    prev_states = (torch.randn([1, 64, 1024]), torch.randn([1, 64, 1024]))
    dec_input = torch.randn([64, 512])
    output, prev_states = test_model(prev_states, dec_input)
    print(output.shape)
        