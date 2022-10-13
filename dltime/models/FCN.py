import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.layers import ConvBlock

class FCN(nn.Module):
    '''FCN'''
    def __init__(self, c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3], clf=True):
        super(FCN, self).__init__()
        self.clf = clf  # 是否作为分类器

        self.convblock1 = ConvBlock(c_in, layers[0], ks=kss[0])
        self.convblock2 = ConvBlock(layers[0], layers[1], ks=kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], ks=kss[2])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return F.softmax(x, dim=-1) if self.clf else x


# if __name__ == "__main__":
#     model = FCN(5, 3)
#     x = torch.randn(64, 5, 96) # bs, chs, seq_len
#     y = model(x)
#     print(y.size())
