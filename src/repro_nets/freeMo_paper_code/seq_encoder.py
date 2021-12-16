import torch
import torch.nn as nn

#由于原始代码中考虑了一个content和style的区分，这里也实现下
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
        #inputs: (T, B, C)
        assert inputs.shape[-1]==self.pose_dim
        S, B = inputs.shape[0], inputs.shape[1]
        outputs, state=self.lstm(inputs)#forward embedding, outputs: (1, batch_size, hidden_size)
        outputs=outputs[-1,:,:]

        # outputs=outputs.squeeze(0)
        assert outputs.shape[0] == B
        outputs=self.fc(outputs)#(batch_size, embed_size)
        
        features=outputs
        # if self.content_dim:
        #     if self.T_layer_norm:
        #         content=outputs[:,0:self.content_dim]
        #         style=self.layer_norm(outputs[:,self.content_dim:])
            
        return features#

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