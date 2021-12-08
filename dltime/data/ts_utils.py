import torch
import numpy as np
import torch.nn as nn


def ucrdf_to_tensor(df):
    '''由sktime导入的UCR/UEA数据集为pandas.Dataframe, 不能直接转为Tensor
    将sktime导入的UCR/UEA数据集转为tentor, 维度为(n, chs, len)
    '''
    ans = []
    for c in df.columns:
        ans.append(torch.Tensor(df[c]).unsqueeze(1))
    
    return torch.cat(ans, dim=1)

def ucrdf_to_nparray(df):
    '''将sktime导入的UCR/UEA数据集转为numpy.array, 维度为(n, chs, len)
    '''
    ans = []
    for c in df.columns:
        ans.append(np.expand_dims(np.array(df[c].tolist()), axis=1))
    
    return np.concatenate(ans, axis=1)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose1d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


