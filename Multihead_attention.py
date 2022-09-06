import torch
from torch import nn
import torch.nn.functional as F
import math
from entmax import entmax15

class Embedding_module(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Embedding_module, self).__init__()
        self.out_ch = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        return self.linear(x) / math.sqrt(self.out_ch)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        attn = entmax15(scores, dim=-1)
        y = torch.matmul(attn, v)
        return self.dropout(y)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, inputs, a_dropout=None, f_dropout=None):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.inputs = inputs
        assert inputs % heads == 0
        self.hidden = inputs // heads

        self.attention = ScaledDotProductAttention(a_dropout)
        self.linear_q = nn.Linear(inputs, inputs)
        self.linear_k = nn.Linear(inputs, inputs)
        self.linear_v = nn.Linear(inputs, inputs)
        self.output = nn.Linear(inputs, inputs)
        self.dropout = nn.Dropout(p=f_dropout) if f_dropout is not None else nn.Identity()

    def forward(self, q, k, v):
        bs = q.size(0)
        q = self.linear_q(q).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        k = self.linear_k(k).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        v = self.linear_v(v).view(bs, -1, self.heads, self.hidden).transpose(1, 2)

        out = self.attention(q, k, v).transpose(1, 2).contiguous()
        out = out.view(bs, -1, self.inputs)
        return self.dropout(self.output(out))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class FeedForwardNet(nn.Module):
    def __init__(self, inputs, hidden, dropout):
        super(FeedForwardNet, self).__init__()
        self.upscale = nn.Linear(inputs, hidden)
        self.activation = Mish()
        self.downscale = nn.Linear(hidden, inputs)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, x):
        x = self.upscale(x)
        x = self.activation(x)
        x = self.downscale(x)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, inputs, heads, hidden, a_dropout=None, f_dropout=None):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(heads, inputs, a_dropout=a_dropout, f_dropout=f_dropout)
        self.attention_norm = nn.LayerNorm(inputs)
        self.feedforward = FeedForwardNet(inputs, hidden, dropout=f_dropout)
        self.feedforward_norm = nn.LayerNorm(inputs)

    def forward(self, x, z=None):
        z = x if z is None else z
        y = self.attention(x, z, z)
        y = self.attention_norm(y)
        x = x + y
        y = self.feedforward(x)
        y = self.feedforward_norm(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, inputs, heads, hidden, a_dropout=None, f_dropout=None):
        super(Encoder, self).__init__()

        self.norm = nn.LayerNorm(inputs)
        self.layers = EncoderLayer(inputs, heads, hidden, a_dropout=a_dropout, f_dropout=f_dropout)

    def forward(self, x, y=None):
        x = self.layers(x, y)
        return self.norm(x)
