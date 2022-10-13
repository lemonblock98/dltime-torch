import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dltime.models.conv_atten import scaled_dot_product_attention, get_pos_encoder, _get_activation_fn
from dltime.models.InceptionTime import InceptionModule
from dltime.base.layers import Conv1dSame, ConvBlock


class InceptionSelfAtten(nn.Module):

    def __init__(self, c_in, c_out=256, d_model=512, num_heads=4):
        super(InceptionSelfAtten, self).__init__()
        assert c_out % (num_heads * 4) == 0

        self.c_in = c_in
        self.c_out = c_out
        self.d_model = d_model
        self.num_heads = num_heads

        nf = c_out // num_heads // 4
        
        # self.wq = Conv1dSame(c_in, c_out, ks=1, stride=1)
        self.wq = nn.ModuleList([InceptionModule(ni=c_in, nf=nf) for _ in range(num_heads)])
        self.wk = nn.ModuleList([InceptionModule(ni=c_in, nf=nf) for _ in range(num_heads)])
        self.wv = nn.ModuleList([InceptionModule(ni=c_in, nf=nf) for _ in range(num_heads)])

        self.final_linear = Conv1dSame(c_out, d_model)

    def forward(self, q, k, v, mask=None):  # q=k=v=x [b, seq_len, embedding_dim] embedding_dim其实也=d_model

        q = torch.cat([conv(q) for conv in self.wq], dim=1)  # =>[bs, d_model, seq_len]
        k = torch.cat([conv(k) for conv in self.wk], dim=1)  # =>[bs, d_model, seq_len]
        v = torch.cat([conv(v) for conv in self.wv], dim=1)  # =>[bs, d_model, seq_len]

        q = torch.cat(q.chunk(self.num_heads, dim=1), dim=0)
        k = torch.cat(k.chunk(self.num_heads, dim=1), dim=0)
        v = torch.cat(v.chunk(self.num_heads, dim=1), dim=0)

        scaled_attention, attention_weights = scaled_dot_product_attention(\
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2), mask)
        # => [b, seq_len_q, d_model], [b, seq_len_q, d_model]

        scaled_attention = torch.cat(scaled_attention.chunk(self.num_heads, dim=0), dim=-1)
        scaled_attention = scaled_attention.transpose(-1, -2)
        output = self.final_linear(scaled_attention)  # =>[b, seq_len_q, d_model=512]
        return output, attention_weights  # [b, d_model, seq_len], [b, seq_len_q, seq_len_k]


class InceptionSelfAttnEncoderLayer(nn.Module):
    def __init__(self, d_in, d_model=512, num_heads=4, dropout=0.1, dim_feedforward=512, activation="relu"):
        super(InceptionSelfAttnEncoderLayer, self).__init__()
        self.dim_feedforward = dim_feedforward
        self.self_attn = InceptionSelfAtten(c_in=d_in, c_out=d_model, d_model=d_model, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps

        self.activation = _get_activation_fn(activation)

        self.shortcut = nn.BatchNorm1d(d_in) if d_in == d_model \
            else ConvBlock(d_in, d_model, ks=1, act=None)
        
        if self.dim_feedforward is not None:
            self.linear1 = nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1, bias=False)
            self.dropout1 = nn.Dropout(dropout)
            self.linear2 = nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1, bias=False)
            self.dropout2 = nn.Dropout(dropout)
            self.shortcut2 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(InceptionSelfAttnEncoderLayer, self).__setstate__(state)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = self.shortcut(src) + self.dropout(src2)  # (batch_size, d_model, seq_len)
        src = self.norm(src)
        if self.dim_feedforward is not None:
            src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
            src = self.shortcut2(src) + self.dropout2(src2)
            src = self.norm2(src)

        return src

class TSInceptionSelfAttnEncoderClassifier(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, num_layers, num_classes, num_heads, dim_feedforward=None, 
                 dropout=0.1, pos_encoding='fixed', activation='gelu', freeze=False):
        super(TSInceptionSelfAttnEncoderClassifier, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.project_inp = nn.Conv1d(in_channels=feat_dim, out_channels=d_model, kernel_size=1, stride=1, bias=False)
        self.encoder = nn.ModuleList([InceptionSelfAttnEncoderLayer(d_model, d_model, num_heads, dim_feedforward=dim_feedforward) for _ in range(num_layers)])
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, X):
        inp = self.project_inp(X) * math.sqrt(
            self.d_model)           # [bs, seq_len, d_model]

        for layer in self.encoder:
            inp = layer(inp)
        
        output = self.act(inp)   # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout1(output)

        # Output
        output = self.output_layer(self.gap(output).squeeze(-1))

        return output


class InceptionOnlyEncoderLayer(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, activation="relu"):
        super(InceptionOnlyEncoderLayer, self).__init__()
        self.self_attn = InceptionModule(c_in=d_model, nf=512 // 4)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(InceptionOnlyEncoderLayer, self).__setstate__(state)

    def forward(self, src):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src)
        src = src + self.dropout(src2)  # (batch_size, d_model, seq_len)
        src = self.norm(src)
        return src

class TSInceptionOnlyClassifier(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, num_layers, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', freeze=False):
        super(TSInceptionOnlyClassifier, self).__init__()

        self.max_len = max_len
        self.d_model = d_model

        self.project_inp = nn.Conv1d(in_channels=feat_dim, out_channels=d_model, kernel_size=1, stride=1, bias=False)
        # self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        self.encoder = nn.ModuleList([InceptionOnlyEncoderLayer(d_model) for _ in range(num_layers)])
        # encoder_layer = (d_model, num_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        # # self.output_layer = nn.Linear(d_model, num_classes)
        # self.output_layer = nn.Linear(d_model * max_len, num_classes)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, X, padding_masks=None):
        """
        Args:
            X: (batch_size, feat_dim, seq_len) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        inp = self.project_inp(X) * math.sqrt(
            self.d_model)           # [bs, d_model, seq_len]

        for layer in self.encoder:
            inp = layer(inp)
        
        output = self.act(inp)   # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout1(output)

        # Output
        output = self.output_layer(self.gap(output).squeeze(-1))

        return output