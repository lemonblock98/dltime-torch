import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dltime.base.layers import Conv1dSame


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    自注意力机制
    - q: query, shape: [..., seq_len_q, depth]
    - k: key, shape: [..., seq_len_k, depth]
    - v: value, shape: [..., seq_len_v, depth_v], seq_len_k == seq_len_v
    有seq_len_q个query, seq_len_k个key, 计算其注意力值及其输出
    """
    # q, k做矩阵乘法, 得到各个query查询各个key得到的value
    matmul_qk = torch.matmul(q, k.transpose(-1, -2)) # [..., seq_len_q, seq_len_k]
    
    # 将得到的value除以sqrt(d_k), 使其不至于太大, 不然输入到softmax后容易导致梯度消失
    dk = torch.tensor(k.shape[-1], dtype=torch.float32) # d_k
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    # 需要 mask 的位置加上一个很大的负值, 使其输入到softmax之后对应概率为0
    if mask is not None:
        scaled_attention_logits += (mask * -1e9).unsqueeze(-2)
    
    # 计算Attention权重矩阵
    attention_weights = F.softmax(scaled_attention_logits, dim=-1) # [..., seq_len_q, seq_len_k]
    
    # 各个value按Attention矩阵加权, 得到各个query对应的最终输出
    output = torch.matmul(attention_weights, v) # [..., seq_len_q, depth_v]
    return output, attention_weights 


class ConvSelfAttention(torch.nn.Module):
    def __init__(self, c_in, c_out=256, d_model=512, num_heads=4, k_size=[1, 3, 5, 7]):
        super(ConvSelfAttention, self).__init__()
        assert c_out % (num_heads * len(k_size)) == 0

        self.c_in = c_in
        self.c_out = c_out
        self.d_model = d_model
        self.num_heads = num_heads

        k_num = c_out // num_heads // len(k_size)
        filter_map = [(ks, k_num) for ks in k_size] * num_heads

        # self.wq = Conv1dSame(c_in, c_out, ks=1, stride=1)
        self.wq = nn.ModuleList([Conv1dSame(c_in, co, ks=ks, stride=1) for ks, co in filter_map])
        self.wk = nn.ModuleList([Conv1dSame(c_in, co, ks=ks, stride=1) for ks, co in filter_map])
        self.wv = nn.ModuleList([Conv1dSame(c_in, co, ks=ks, stride=1) for ks, co in filter_map])

        self.final_linear = nn.Linear(c_out, d_model)

    def forward(self, q, k, v, mask):  # q=k=v=x [b, seq_len, embedding_dim] embedding_dim其实也=d_model

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
        output = self.final_linear(scaled_attention)  # =>[b, seq_len_q, d_model=512]
        return output.transpose(-1, -2), attention_weights  # [b, d_model, seq_len], [b, seq_len_q, seq_len_k]


class TransformerConvAttnBNEncoderLayer(nn.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model=512, num_heads=8, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(TransformerConvAttnBNEncoderLayer, self).__init__()
        self.self_attn = ConvSelfAttention(c_in=d_model, c_out=256, d_model=d_model, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)

        self.dim_feedforward = dim_feedforward
        if self.dim_feedforward is not None:
            self.linear1 = nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1, bias=False)
            self.linear2 = nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1, bias=False)
            self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerConvAttnBNEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        # src = src.permute(0, 2, 1)  # restore (batch_size, seq_len, d_model)
        if self.dim_feedforward is not None:
            src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)  # (batch_size, d_model, d_model)
            # src = src.permute(0, 2, 1)
            src = self.norm2(src)       # (batch_size, d_model, seq_len)
        return src


class TSTransformerEncoderConvClassifier(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, num_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', freeze=False):
        super(TSTransformerEncoderConvClassifier, self).__init__()

        self.max_len = max_len
        self.d_model = d_model

        self.project_inp = nn.Conv1d(in_channels=feat_dim, out_channels=d_model, kernel_size=1, stride=1, bias=False)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        encoder_layer = TransformerConvAttnBNEncoderLayer(d_model, num_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
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
        # inp = X.permute(0, 2, 1)    # [bs, seq_len, d_in]
        inp = self.project_inp(X) * math.sqrt(
            self.d_model)           # [bs, seq_len, d_model]
        inp = inp.permute(2, 0, 1)  # [seq_len, bs, d_model]
        inp = self.pos_enc(inp)     # add positional encoding
        inp = inp.permute(1, 2, 0)  # [bs, d_model, seq_len]

        output = self.transformer_encoder(inp, src_key_padding_mask=padding_masks)  # (batch_size, d_model, seq_length)
        output = self.act(output)   # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout1(output)

        # Output
        # output = output.view(-1, self.d_model * self.max_len)
        # output = self.output_layer(output)  # (batch_size, num_classes)
        output = self.output_layer(self.gap(output).squeeze())

        return output


class ConvAttenEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(ConvAttenEncoderLayer, self).__init__()
        self.self_attn = ConvSelfAttention(c_in=d_model, c_out=256, d_model=d_model, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(ConvAttenEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)  # (batch_size, d_model, seq_len)
        src = self.norm(src)
        return src



class TSConvAttenClassifier(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, num_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', freeze=False):
        super(TSConvAttenClassifier, self).__init__()

        self.max_len = max_len
        self.d_model = d_model

        self.project_inp = nn.Conv1d(in_channels=feat_dim, out_channels=d_model, kernel_size=1, stride=1, bias=False)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        self.encoder = nn.ModuleList([ConvAttenEncoderLayer(d_model, num_heads, dim_feedforward) for _ in range(num_layers)])
        # encoder_layer = (d_model, num_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        # self.output_layer = nn.Linear(d_model, num_classes)
        self.output_layer = nn.Linear(d_model * max_len, num_classes)

    def forward(self, X, padding_masks=None):
        """
        Args:
            X: (batch_size, feat_dim, seq_len) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        # inp = X.permute(0, 2, 1)    # [bs, seq_len, d_in]
        inp = self.project_inp(X) * math.sqrt(
            self.d_model)           # [bs, seq_len, d_model]
        inp = inp.permute(2, 0, 1)  # [seq_len, bs, d_model]
        inp = self.pos_enc(inp)     # add positional encoding
        inp = inp.permute(1, 2, 0)  # [bs, d_model, seq_len]

        for layer in self.encoder:
            inp = layer(inp, src_key_padding_mask=padding_masks)
        
        output = self.act(output)   # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout1(output)

        # Output
        # gap_weight = F.softmax(padding_masks * -1e9, dim=-1).unsqueeze(-1)
        # output = torch.bmm(output, gap_weight).squeeze()
        output = output.view(-1, self.d_model * self.max_len)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output

