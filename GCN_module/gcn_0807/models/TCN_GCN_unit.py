import torch
from torch import nn
import numpy as np
import math
from math import sqrt
from entmax import entmax15
import torch.nn.functional as F

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class conv_residual(nn.Module):
    '''the residual of gcn and temporal attention module
        to adjust channels by using convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(conv_residual, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        y = self.conv(x)
        return self.bn(y)


class decoupling_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive) -> None:
        super(decoupling_gcn, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.subnet = A.shape[0]
        self.groups = 8
        self.adaptive = adaptive

        self.linear = nn.Linear(self.in_ch, self.out_ch)

        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
            3, 1, 19, 19]), dtype=torch.float32, requires_grad=True).repeat(1, self.groups, 1, 1), requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * self.subnet)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * self.subnet, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * self.subnet)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * self.subnet, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant_(self.Linear_bias, 1e-6)

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        learn_A = self.DecoupleA.repeat(1, self.out_ch // self.groups, 1, 1)  # learn_A -> [3, 32, 19, 19]
        norm_learn_A = torch.cat([self.L2_norm(learn_A[0:1, ...]),
                                  self.L2_norm(learn_A[1:2, ...]),
                                  self.L2_norm(learn_A[2:3, ...])],
                                 0)
        # norm_A -> [3, 32, 19, 19]
        y = torch.einsum('nctw,cd->ndtw', (x, self.Linear_weight)).contiguous()
        y = y + self.Linear_bias
        y = self.bn0(y)

        n, kc, t, v = y.size()
        y = y.view(n, self.subnet, kc // self.subnet, t, v)
        y = torch.einsum('nkctv,kcvw->nctw', (y, norm_learn_A))

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y


class Embedding_module(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Embedding_module, self).__init__()
        self.out_ch = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        return self.linear(x) / sqrt(self.out_ch)


class Encoder(nn.Module):
    def __init__(self, inputs, heads, hidden, a_dropout=None, f_dropout=None):
        '''Implemented encoder via multiple stacked encoder layers'''
        super(Encoder, self).__init__()

        self.norm = nn.LayerNorm(inputs)
        self.layers = EncoderLayer(inputs, heads, hidden, a_dropout=a_dropout, f_dropout=f_dropout)

    def forward(self, x, mask=None):
        x = self.layers(x, mask)
        return self.norm(x)  # x


class EncoderLayer(nn.Module):
    def __init__(self, inputs, heads, hidden, a_dropout=None, f_dropout=None):
        '''Implemented encoder layer via multi-head self-attention and feedforward net'''
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(heads, inputs, a_dropout=a_dropout, f_dropout=f_dropout)
        self.attention_norm = nn.LayerNorm(inputs)
        self.feedforward = FeedForwardNet(inputs, hidden, dropout=f_dropout)
        self.feedforward_norm = nn.LayerNorm(inputs)

    def forward(self, x, mask=None):
        y = self.attention_norm(x)
        y = self.attention(y, y, y, mask=mask)
        x = x + y
        y = self.feedforward_norm(x)
        y = self.feedforward(y)
        x = x + y
        return x


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

    def forward(self, q, k, v, mask):
        bs = q.size(0)
        q = self.linear_q(q).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        k = self.linear_k(k).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        v = self.linear_v(v).view(bs, -1, self.heads, self.hidden).transpose(1, 2)

        out = self.attention(q, k, v, mask).transpose(1, 2).contiguous()
        out = out.view(bs, -1, self.inputs)
        return self.dropout(self.output(out))


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout=None):
        '''Implemented simple attention'''
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
        self.attn_type = 'entmax15'

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e12)
        attn = entmax15(scores, dim=-1)
        return torch.matmul(attn, v)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish avtivation loaded...")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class FeedForwardNet(nn.Module):
    def __init__(self, inputs, hidden, dropout):
        '''Implemented feedforward network'''
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

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.feature_dims = out_channels * 3
        self.num_heads = 8
        self.hidden_dims = out_channels * 6
        self.attention_dropout = 0.3
        self.tat_times = 5

        self.res = residual
        self.gcn = decoupling_gcn(in_channels, out_channels, A, adaptive)

        self.embed = Embedding_module(out_channels, self.feature_dims)
        self.tat = nn.ModuleList(
            Encoder(self.feature_dims, self.num_heads, self.hidden_dims, self.attention_dropout) for _ in
            range(self.tat_times))
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            #self.residual = lambda x: x
            self.residual = conv_residual(in_channels, out_channels, kernel_size=1, stride=stride)

        self.linear = nn.Linear(self.feature_dims, out_channels)

    def forward(self, x):  # x->[2, 4, T, 19]
        #x = self.embed(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        gcn = self.gcn(x)
        res = self.residual(x)
        x = gcn + res
        B, C, T, V = x.size()
        tcn = self.embed(x.permute(0, 3, 2, 1)).contiguous().view(B*V, T, -1)
        #tcn = x.permute(0, 3, 2, 1).contiguous().view(B*V, T, -1)
        for i in range(self.tat_times):
            memory = tcn
            tcn = self.tat[i](tcn)
            tcn += memory
        y = self.linear(tcn).contiguous().view(B, -1, T, V)
        y = self.relu(y)
        return y
