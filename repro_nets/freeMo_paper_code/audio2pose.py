import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, channels, padding=3, kernel_size=8, conv_stride=2, conv_pool=None, augmentation=False):
        super(AudioEncoder, self).__init__()
        self.in_channels = channels[0]
        self.augmentation = augmentation

        model = []
        acti = nn.LeakyReLU(0.2)

        nr_layer = len(channels)-1

        for i in range(nr_layer):
            if conv_pool is None:
                model.append(nn.ReflectionPad1d(padding))
                model.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernel_size, stride=conv_stride))
                model.append(acti)
            else:
                model.append(nn.ReflectionPad1d(padding))
                model.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernel_size, stride=conv_stride))
                model.append(acti)
                model.append(conv_pool(kernel_size=2, stride=2))

        if self.augmentation:
            model.append(
                nn.Conv1d(channels[-1], channels[-1], kernel_size=kernel_size, stride=conv_stride)
            )
            model.append(acti)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #x:batch_size, 13, 100
        x = x[:, :self.in_channels, :]
        x = self.model(x)
        return x

class AudioDecoder(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(AudioDecoder, self).__init__()

        model = []
        pad = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)
        
        for i in range(len(channels) - 2):
            model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            model.append(nn.ReflectionPad1d(pad))
            model.append(nn.Conv1d(channels[i], channels[i + 1],
                                            kernel_size=kernel_size, stride=1))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=0.2))
            if not i == len(channels) - 2:
                model.append(acti)          
        
        model.append(nn.Upsample(size=25, mode='nearest'))
        model.append(nn.ReflectionPad1d(pad))
        model.append(nn.Conv1d(channels[-2], channels[-1],
                                            kernel_size=kernel_size, stride=1))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Audio2Pose(nn.Module):
    def __init__(self, pose_dim, embed_size, augmentation):
        super(Audio2Pose, self).__init__()
        self.pose_dim = pose_dim
        self.embed_size = embed_size
        self.augmentation = augmentation

        self.aud_enc = AudioEncoder(channels=[13,64,128,256], padding=2, kernel_size=7, conv_stride=1, conv_pool=nn.AvgPool1d, augmentation = self.augmentation)
        if self.augmentation:
            self.aud_dec = AudioDecoder(channels=[512, 256, 128, pose_dim])
        else:
            self.aud_dec = AudioDecoder(channels=[256, 256, 128, pose_dim])

        if self.augmentation:
            self.pose_enc = nn.Sequential(
                nn.Linear(self.embed_size // 2, 256),
                nn.LayerNorm(256)
            )

    def forward(self, audio_feat, dec_input = None):
        #audio_feat: (B, C, T)
        B = audio_feat.shape[0]

        aud_embed = self.aud_enc.forward(audio_feat)#(B, 256, 4)

        if self.augmentation:
            dec_input = dec_input.squeeze(0) #(B, embed_size // 2)
            dec_embed = self.pose_enc(dec_input) #(B, 256)
            dec_embed = dec_embed.unsqueeze(2) #(B, 256, 1)
            dec_embed = dec_embed.expand(dec_embed.shape[0], dec_embed.shape[1], aud_embed.shape[-1])
            aud_embed = torch.cat([aud_embed, dec_embed], dim=1) #(B, 512, 1)

        out = self.aud_dec.forward(aud_embed) # 
        return out

if __name__ == '__main__':
    test_model = Audio2Pose(
        pose_dim=108,
        embed_size=256,
        augmentation=False
    )

    dummy_input = torch.randn(64, 13, 100)
    output = test_model(dummy_input, None)

    print(output.shape)