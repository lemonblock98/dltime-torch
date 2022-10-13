import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """MLP"""
    def __init__(self, c_in, c_out, seq_len, ps=[0.1, 0.2, 0.2, 0.3], layers=[500, 500, 500]):
        super().__init__()
        assert len(ps) == len(layers) + 1   # dropout层与linear层数量应保持一致
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1) # 若输入为多维序列则flatten
        self.nfs = [c_in * seq_len] + layers + [c_out]     # 各层的特征数
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(ps[i]),
                nn.Linear(self.nfs[i], self.nfs[i+1]),
                nn.ReLU()
            ) for i in range(len(ps))])
    
    def forward(self, x):
        x = self.flatten(x)
        for l in self.linears:
            x = l(x)
        
        return F.softmax(x, dim=-1)


# if __name__ == "__main__":
#     model = MLP(1, 3, 96)
#     x = torch.randn(64, 96)
#     y = model(x)
#     print(y.size())