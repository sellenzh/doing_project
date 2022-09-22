import math
import torch
from torch import nn
import numpy as np

import torch.nn.functional as F
from entmax import entmax15

def ConvLayer(in_channels, out_channels, dims=1,
              kernel_size=1, stride=1, padding=0, bias=True):
    if dims == 1:
        layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias)
    elif dims == 2:
        layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias)
    return layer

class Conv2BN2Silu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, bias=True):
        unit = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias)]
        unit.append(nn.BatchNorm2d(out_channels))
        unit.append(nn.SiLU())
        super(Conv2BN2Silu, self).__init__(*unit)


class Conv1BN1Silu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, bias=True):
        unit = [nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias)]
        unit.append(nn.BatchNorm1d(out_channels))
        unit.append(nn.SiLU())
        super(Conv1BN1Silu, self).__init__(*unit)


class GCN_TAT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, args,
                 stride=1, residual=True, adaptive=True):
        super(GCN_TAT_unit, self).__init__()
        self.feature_dims = out_channels * 3
        self.head_num = args.head_num
        self.hidden_dims = out_channels * 6
        self.rate = args.dropout_rate
        self.tat_num = args.layer_num
        self.res = residual

        self.gcn = DecouplingGCN(in_channels, out_channels,
                                 args, adaptive=False)
        self.embed = Embedding(out_channels, self.feature_dims)

        self.time_att = nn.ModuleList(
            Encoder(inputs=self.feature_dims, hidden=self.hidden_dims, args=args)
            for _ in range(self.tat_num))
        self.relu = nn.ReLU()

        
        



def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class DecouplingGCN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 args, adaptive=True):
        super(DecouplingGCN, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.groups = args.groups
        self.nodes = args.nodes
        self.device = args.device
        self.adaptive = adaptive
        self.A = np.stack([np.eye(self.nodes)] * 3, axis=0)
        self.subnet = self.A.shape[0]

        self.linear = nn.Linear(self.in_ch, self.out_ch)
        self.DecoupleA = nn.Parameter(torch.tensor(
            np.reshape(self.A.astype(np.float32), [3, 1, 19, 19]), dtype=torch.float32,
            requires_grad=True).repeat(1, self.groups, 1, 1), requires_grad=True)

        if self.in_ch != self.out_ch:
            self.down_weight = nn.Parameter(torch.zeros(
                self.in_ch, self.out_ch, requires_grad=True,
                device=self.device), requires_grad=True)
            nn.init.normal_(self.down_weight, 0, math.sqrt(
                0.5 / (self.out_ch * self.subnet)
            ))
            self.down_bias = nn.Parameter(torch.zeros(
                1, self.out_ch * self.subnet, 1, 1, requires_grad=True,
                device=self.device), requires_grad=True)
            nn.init.constant_(self.down_bias, 1e-6)

        self.bn0 = nn.BatchNorm2d(self.out_ch * self.subnet)
        self.bn1 = nn.BatchNorm2d(self.out_ch)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            self.in_ch, self.out_ch * self.subnet, requires_grad=True, device=self.device), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (self.out_ch * self.subnet)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, self.out_ch * self.subnet, 1, 1, requires_grad=True, device=self.device), requires_grad=True)
        nn.init.constant_(self.Linear_bias, 1e-6)

    def L2_norm(self, A):
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4
        A = A / A_norm
        return A

    def forward(self, x):
        learn_A = self.DecoupleA.repeat(1, self.out_ch // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.L2_norm(learn_A[0:1, ...]),
                                  self.L2_norm(learn_A[1:2, ...]),
                                  self.L2_norm(learn_A[2:3, ...])], 0)

        y = torch.einsum('nctv,cd->ndtv', (x, self.Linear_weight)).contiguous()
        y += self.Linear_bias
        y = self.bn0(y)

        n, kc, t, v = y.size()
        y = y.view(n, self.subnet, kc // self.subnet, t, v)
        y = torch.einsum('nkctv,kcvw->nctw', (y, norm_learn_A))
        y = self.bn1(y)

        if self.in_ch != self.out_ch:
            down = torch.einsum('nctv,cd->ndtv', (x, self.down_weight)).contiguous()
            down += self.down_bias
            down = self.bn1(down)
        else:
            down = x

        y += down
        return self.relu(y)


class Embedding(nn.Module):
    def __init__(self, in_channel, out_channels, bias=False):
        super(Embedding, self).__init__()
        self.out_ch = out_channels
        self.linear = nn.Linear(in_channel, out_channels, bias=bias)
        
    def forward(self, x):
        return self.linear(x) / math.sqrt(self.out_ch)


class Encoder(nn.Module):
    def __init__(self, inputs, hidden, args):
        super(Encoder, self).__init__()

        self.norm = nn.LayerNorm(inputs)
        self.layers = EncoderLayers(inputs, hidden, args)

    def forward(self, x):
        y = self.layers(x)
        return self.norm(y)


class EncoderLayers(nn.Module):
    def __init__(self, inputs, hidden, args):
        super(EncoderLayers, self).__init__()
        self.inputs = inputs
        self.hidden = hidden
        self.rate = args.dropout_rate
        self.heads = args.head_num

        self.attention = MultiHeadAttention(self.heads, self.inputs, self.rate)
        self.attention_norm = nn.LayerNorm(self.inputs)

        self.feedforward = FeedForwardNet(self.inputs, self.hidden, self.rate)
        self.feedforward_norm = nn.LayerNorm(self.inputs)

    def forward(self, x):
        y = self.attention_norm(x)
        y = self.attention(y, y, y)
        x += y
        y = self.feedforward_norm(x)
        y = self.feedforward(y)
        x += y
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, inputs, rate):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.inputs = inputs
        assert inputs % heads == 0
        self.hidden = inputs // heads

        self.attention = ScaledDotProductAttention(self.rate)
        self.linear_q = nn.Linear(inputs, inputs)
        self.linear_k = nn.Linear(inputs, inputs)
        self.linear_v = nn.Linear(inputs, inputs)

        self.output = nn.Linear(inputs, inputs)
        self.dropout = nn.Dropout(rate) if rate is not None else nn.Identity()

    def forward(self, q, k, v):
        bs = q.size(0)
        q_w = self.linear_q(q).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        k_w = self.linear_k(k).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        v_w = self.linear_v(v).view(bs, -1, self.heads, self.hidden).transpose(1, 2)

        out = self.attention(q_w, k_w, v_w).transpose(1, 2).contiguous()
        out = out.view(bs, -1, self.inputs)
        return self.dropout(self.output(out))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, rate=None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(rate) if rate is not None else nn.Identity()
        self.attn_type = 'entmax15'

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        att = entmax15(scores, dim=-1)
        return torch.matmul(att, v)


class FeedForwardNet(nn.Module):
    def __init__(self, inputs, hidden, rate=None):
        super(FeedForwardNet, self).__init__()
        self.upscale = nn.Linear(inputs, hidden)
        self.activation = Mish()
        self.downscale = nn.Linear(hidden, inputs)
        self.dropout = nn.Dropout(rate) if rate is not None else nn.Identity()

    def forward(self, x):
        y = self.upscale(x)
        y = self.activation(y)
        y = self.downscale(y)
        return self.dropout(y)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))
