import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.layers import ConvBlock

class ResBlock(nn.Module):
    "ResBlock"
    def __init__(self, ni, nf, kss=[8, 5, 3]):
        super().__init__()

        self.convblocks = nn.Sequential(
            ConvBlock(ni, nf, kss[0]),
            ConvBlock(nf, nf, kss[1]),
            ConvBlock(nf, nf, kss[2], act=None)
        )
        
        self.shortcut = nn.BatchNorm1d(ni) if ni == nf else \
            ConvBlock(ni, nf, 1, act=None)
        
    def forward(self, x):
        res = x
        x = self.convblocks(x)
        x = x + self.shortcut(res)
        return F.relu(x, inplace=True)


class ResNet(nn.Module):
    "ResNet"
    def __init__(self, c_in, c_out, layers=[64, 128, 128]):
        super().__init__()
        self.resblocks = nn.Sequential(
            ResBlock(c_in, layers[0]),
            ResBlock(layers[0], layers[1]),
            ResBlock(layers[1], layers[2]),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(layers[-1], c_out)
    
    def forward(self, x):
        x = self.resblocks(x)   # (bs, layers[-1], seq_len)
        x = self.gap(x).squeeze(-1) # (bs, layers[-1], 1)  -> (bs, layers[-1])
        return F.softmax(self.fc(x), dim=-1)


# if __name__ == "__main__":
#     model = ResNet(5, 3)
#     x = torch.randn(64, 5, 96) # bs, chs, len
#     y = model(x)
#     print(y.size())

