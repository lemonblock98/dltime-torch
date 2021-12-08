import torch
import torch.nn as nn
from data.ts_utils import initialize_weights


class TSDCGenerator(nn.Module):
    def __init__(self, sig_len=64, latent_dim=64, output_channel=1):
        super(TSDCGenerator, self).__init__()
        self.seq_len = sig_len
        self.latent_dim = latent_dim
        self.output_channel = output_channel
        
        self.init_size = sig_len // 8
        # 相当于除了三次2，因此在反卷积中要乘三次二变回来
        
        # fc: Linear -> BN -> ReLU
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size),
            nn.BatchNorm1d(512 * self.init_size),
            nn.ReLU(inplace=True)
        )
        
        # deconv: ConvTranspose2d(4, 2, 1) -> BN -> ReLU -> 
        #         ConvTranspose2d(4, 2, 1) -> BN -> ReLU -> 
        #         ConvTranspose2d(4, 2, 1) -> Tanh
        # 根据o = s(i-1)-2p+k，o = s(i)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, output_channel, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 512, self.init_size)
        img = self.deconv(out)
        return img


class TSDCDiscriminator(nn.Module):
    def __init__(self, sig_len=32, input_channel=1, sigmoid=True):
        super(TSDCDiscriminator, self).__init__()
        self.image_size = sig_len
        self.input_channel = input_channel
        self.fc_size = sig_len // 8
        
        # conv: Conv2d(3,2,1) -> LeakyReLU 
        #       Conv2d(3,2,1) -> BN -> LeakyReLU 
        #       Conv2d(3,2,1) -> BN -> LeakyReLU 
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, 2, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 3, 2, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )
        
        # fc: Linear -> Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(512 * self.fc_size, 1),
        )

        if sigmoid:
            self.fc.add_module('sigmoid', nn.Sigmoid())
        initialize_weights(self)


    def forward(self, img):
        out = self.conv(img)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out


# if __name__ == '__main__':
#     G = TSDCGenerator(64, latent_dim=100, output_channel=3)
#     D = TSDCDiscriminator(64, input_channel=3)
#     z = torch.rand(20, 100)
#     g_z = G(z)
#     print('Generate Z:', g_z.size())
#     print('Fake digit:', D(g_z).size())

