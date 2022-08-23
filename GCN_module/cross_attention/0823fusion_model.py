import torch
from torch import nn
import math

class Fusion_model(nn.Module):
    def __init__(self, inputs):
        super(Fusion_model, self).__init__()
        self.attention = MultiHeadAttention(inputs)

    def forward(self, pose, vel):
        '''pose :[B, C, T', V], vel : [B, C, T]'''
        b, c, t, v = pose.size()
        pose = pose.permute(0, 2, 3, 1).contiguous().view(-1, v, c) #[b*t, v, c]
        vel = vel.permute(0, 2, 1).contiguous().view(-1, c).unsqueeze(-2)#[b*t, 1, c]
        y = self.attention(pose, vel)
        y = y.contiguous().view(b, t, v, c).permute(0, 3, 1, 2)
        return y


class AttentionWeigths(nn.Module):

    def __init__(self, dropout=None):
        '''Implemented simple attention'''
        super(AttentionWeigths, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = nn.Softmax(dim=-1)(scores)
        return torch.matmul(attn, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, inputs, a_dropout=None, f_dropout=None):
        '''Implemented simple multi-head attention'''
        super(MultiHeadAttention, self).__init__()
        self.inputs = inputs
        self.hiddens = inputs * 2

        self.attention = AttentionWeigths(a_dropout)
        self.linear_q = nn.Linear(inputs, self.hiddens)
        self.linear_k = nn.Linear(inputs, self.hiddens)
        self.linear_v = nn.Linear(inputs, self.hiddens)
        self.output = nn.Linear(self.hiddens, inputs)
        self.dropout = nn.Dropout(p=f_dropout) if f_dropout is not None else nn.Identity()

    def forward(self, q, k):
        v = k
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        out = self.attention(q, k, v)
        return self.dropout(self.output(out))

'''import random
T = random.randint(2, 62)
pose = torch.randn(size=(16, 32, T, 19))
vel = torch.randn(size=(16, 32, T))
pose2 = torch.randn(size=(16, 64, T, 19))
vel2 = torch.randn(size=(16, 64, T))
model1 = Fusion_model(32)
model2 = Fusion_model(64)
y1 = model1(pose, vel)
y2 = model2(pose2, vel2)
print(y1.size(), y2.size())'''
