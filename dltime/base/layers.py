import torch
import torch.nn as nn
import torch.nn.functional as F


def same_padding1d(seq_len, ks, stride=1, dilation=1):
    '''
    与Tensorflow一致的same padding策略(一维卷积)\n
    Args:
    - seq_len: 输入序列长度
    - ks: 卷积核大小
    '''
    p = (seq_len - 1) * stride + (ks - 1) * dilation + 1 - seq_len
    return p // 2, p - p // 2


class Conv1dSame(nn.Module):
    "Conv1d with padding='same'"
    def __init__(self, ni, nf, ks=3, stride=1, dilation=1, **kwargs):
        super(Conv1dSame, self).__init__()
        self.ks, self.stride, self.dilation = ks, stride, dilation
        self.conv1d = nn.Conv1d(ni, nf, ks, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        # x shape: (bs, ch, seq_len)
        self.padding = same_padding1d(x.shape[-1], self.ks, dilation=self.dilation)
        return self.conv1d(nn.ConstantPad1d(self.padding, value=0.)(x))


class ConvBlock(nn.Module):
    "Conv + BN + Act"
    def __init__(self, ni, nf, ks=3, stride=1, dilation=1, padding='same', act=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        self.act = act # 是否使用激活函数
        self.conv1d = Conv1dSame(ni, nf, ks, stride, dilation) if padding=='same' else \
            nn.Conv1d(ni, nf, ks, stride=stride, dilation=dilation)
        # 1d卷积
        
        self.bn = nn.BatchNorm1d(nf) # BN层
    
    def forward(self, x):
        y = self.bn(self.conv1d(x))
        if self.act is not None:
            return self.act(y)
        else:
            return y


def same_padding2d(H, W, ks, stride=(1,1), dilation=(1,1)):
    '''
    二维卷积samepadding策略
    '''
    p_h = 0 if ks[0] == 1 else (H - 1) * stride[0] + (ks[0] - 1) * dilation[0] + 1 - H
    p_w = 0 if ks[1] == 1 else (W - 1) * stride[1] + (ks[1] - 1) * dilation[1] + 1 - W

    return (p_h // 2, p_h - p_h // 2), (p_w // 2, p_w - p_w // 2)


class SqueezeExciteBlock(nn.Module):
    "SE Block"

    def __init__(self, ni, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ni, ni // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ni // reduction, ni, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.gap(x).squeeze() # bs, chs
        y = self.fc(y).unsqueeze(2) # bs, chs, 1
        return x * y.expand_as(x) # bs, chs, seq_len


# if __name__ == "__main__":
#     conv1dsame = Conv1dSame(3, 128, ks=12)
#     x = torch.randn(64, 3, 96)
#     y = conv1dsame(x)
#     print(y.size())
