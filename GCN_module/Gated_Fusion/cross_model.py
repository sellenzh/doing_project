import torch
from torch import nn
import math
from entmax import entmax15


class CrossTransformer(nn.Module):
    def __init__(self, inputs):
        super(CrossTransformer, self).__init__()
        self.heads = 8
        self.attention = MultiHeadAttention(self.heads, inputs)

    def forward(self, x, y):
        #x=[B, C, T, V]
        #y=[B, C, T]
        B, C, T, V = x.size()
        p = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, -1) # x->[B*T, V, C]
        y = y.permute(0, 2, 1)
        y = y.contiguous().view(B * T, C)
        y = y.unsqueeze(-2) # y->[B*T, 1, C]
        z = self.attention(p, y, y).view(B, T, V, -1).permute(0, 3, 1, 2)
        z += x
        return z

class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout=None):
        '''Implemented simple attention'''
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = entmax15(scores, dim=-1)
        y = torch.matmul(attn, v)
        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, inputs, a_dropout=None, f_dropout=None):
        '''Implemented simple multi-head attention'''
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
        y = self.dropout(self.output(out))
        y = y.view(bs, -1, )
        return y

'''model = CrossTransformer(inputs=32)
model2 = CrossTransformer(inputs=64)
import random
T = random.randint(2, 62)
tensor1 = torch.randn(size=(16, 32, T, 19))
tensor2 = torch.randn(size=(16, 64, T, 19))
spe1 = torch.randn(size=(16, 32, T))
spe2 = torch.randn(size=(16, 64, T))
f1 = model(tensor1, spe1)
f2 = model2(tensor2, spe2)
print(f1.size(), f2.size())'''
