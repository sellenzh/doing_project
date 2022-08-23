import torch
from torch import nn
import math
from entmax import entmax15


class CrossTransformer(nn.Module):
    def __init__(self, inputs):
        super(CrossTransformer, self).__init__()
        self.input = inputs

        self.expend = nn.Sequential(
            nn.Linear(1, self.input), nn.ReLU()
        )
        self.bn = nn.BatchNorm2d(self.input)
        self.squeeze = nn.Sequential(
            nn.Linear(self.input, 19),nn.ReLU()
        )
        self.attention = MultiHeadAttention(self.input)

    def forward(self, pose, vel):
        memory = pose
        _, c, t, v = pose.size()
        vel = self.expend(vel.unsqueeze(-1))
        vel = self.bn(vel)
        vel = self.squeeze(vel).permute(0, 3, 2, 1).contiguous().view(-1, t, c)
        pose = pose.permute(0, 3, 2, 1).contiguous().view(-1, t, c)

        z = self.attention(pose, vel)
        z = z.contiguous().view(-1, v, t, c).permute(0, 3, 2, 1)
        z += memory
        return z

class ModalityAttention(nn.Module):
    def __init__(self):
        super(ModalityAttention, self).__init__()
        self.softmax = nn.Softmax(0)

    def forward(self, w1, f1, w2, f2):
        T = w1.size(-2)
        weights = torch.cat((w1.unsqueeze(0), w2.unsqueeze(0)), dim=0)
        weights, idx = torch.max(weights, dim=-1)
        weights = self.softmax(weights).unsqueeze(-1).repeat(1, 1, 1, T)#[B*V, T, 1]
        res1 = torch.matmul(weights[0, :], f1)
        res2 = torch.matmul(weights[1, :], f2)
        y = res1 + res2
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

'''model = CrossTransformer(inputs=32)
model2 = CrossTransformer(inputs=64)
import random
T = random.randint(2, 62)
tensor1 = torch.randn(size=(16, 32, T, 19))
tensor2 = torch.randn(size=(16, 64, T, 19))
spe1 = torch.randn(size=(16, 32, T))
spe2 = torch.randn(size=(16, 64, T))
y = model(tensor1, spe1)
y2 = model2(tensor2, spe2)
print(y.size())
print(y2.size())'''
