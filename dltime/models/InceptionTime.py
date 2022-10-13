import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.layers import Conv1dSame, ConvBlock


class InceptionModule(nn.Module):
    "InceptionModule"

    def __init__(self, ni, nf=32, ks=40, bottleneck=True):
        super().__init__()
        kss = [ks // (2 ** i) for i in range(3)]            # kss = [10, 20, 40]
        kss = [ks if ks % 2 != 0 else ks - 1 for ks in kss] # kss = [9, 19, 39]
        bottleneck = bottleneck if ni > 1 else False        # 单维序列则不需要 bottleneck
        
        self.bottleneck = Conv1dSame(ni, nf, ks=1, bias=False) if bottleneck else lambda x: x
        self.convs = nn.ModuleList([
            Conv1dSame(nf if bottleneck else nf, nf, ks, bias=False) for ks in kss
        ]) # 3种维度的卷积
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            Conv1dSame(ni, nf, ks=1, bias=False)
        ) # 步长为3的最大值池化 + 1d卷积

        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(x) # 多维序列需要经过 bottleneck 层
        # 平行经过3层卷积, 一层最大值池化
        x = torch.cat([conv(x) for conv in self.convs] + [self.maxconvpool(input_tensor)], dim=1)
        return self.act(self.bn(x))


class InceptionBlock(nn.Module):
    "InceptionBlock"

    def __init__(self, ni, nf=32, residual=True, **kwargs):
        super().__init__()
        self.residual = residual

        self.incepmodules = nn.Sequential(
            InceptionModule(ni, nf, **kwargs),
            InceptionModule(nf*4, nf, **kwargs),
            InceptionModule(nf*4, nf, **kwargs)
        )

        if residual:
            n_in, n_out = ni, nf * 4
            self.shortcut = nn.BatchNorm1d(n_in) if n_in == n_out \
                else ConvBlock(n_in, n_out, ks=1, act=None)
        
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        res = x
        x = self.incepmodules(x)
        x = x + self.shortcut(res) if self.residual else x
        return self.act(x)


class InceptionTime(nn.Module):
    "InceptionTime"

    def __init__(self, c_in, c_out, nf=32):
        super().__init__()
        self.inceptionblock1 = InceptionBlock(c_in, nf)
        self.inceptionblock2 = InceptionBlock(nf*4, nf)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nf * 4, c_out)

    def forward(self, x):
        x = self.inceptionblock1(x)
        x = self.inceptionblock2(x)
        x = self.gap(x).squeeze(-1)
        return F.softmax(self.fc(x), dim=-1)


# if __name__ == "__main__":
#     model = InceptionTime(c_in=12, c_out=5)
#     x = torch.randn([64, 12, 96]) # bs, chs, seq_len
#     y = model(x)
#     print(y.size())
