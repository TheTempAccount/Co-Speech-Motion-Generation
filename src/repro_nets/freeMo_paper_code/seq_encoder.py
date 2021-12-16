import torch
import torch.nn as nn


class SeqEncoder(nn.Module):
    def __init__(self, 
        pose_dim, 
        embed_size, 
        content_dim,
        T_layer_norm, 
        hidden_size, 
        device
    ):
        '''
        LSTM - FC
        '''
        super(SeqEncoder, self).__init__()

        self.pose_dim=pose_dim
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.device=device
        self.content_dim = content_dim
        self.T_layer_norm = T_layer_norm
        self.lstm=nn.LSTM(input_size=pose_dim, hidden_size=self.hidden_size, num_layers=1)
        self.fc=nn.Linear(self.hidden_size, self.embed_size)

        if content_dim > 0:
            if T_layer_norm>0 :
                self.layer_norm=nn.LayerNorm(self.embed_size//2)


    def forward(self, inputs):
        
        assert inputs.shape[-1]==self.pose_dim
        S, B = inputs.shape[0], inputs.shape[1]
        outputs, state=self.lstm(inputs)
        outputs=outputs[-1,:,:]

        
        assert outputs.shape[0] == B
        outputs=self.fc(outputs)
        
        features=outputs
        
        
        
        
            
        return features

if __name__ == "__main__":
    test_model = SeqEncoder(
        108,
        1024,
        512,
        True,
        1024,
        device=torch.device('cpu')
    )

    inputs = torch.randn([25, 64, 108])
    outputs = test_model(inputs)
    print(outputs.shape)