import math
import torch
from torch import nn
import numpy as np
from math import sqrt

import torch.nn.functional as F
from entmax import entmax15


class pedMondel(nn.Module):

    def __init__(self, frames, vel=False, seg=False, h3d=True, nodes=19, n_clss=1):
        super(pedMondel, self).__init__()
        self.frames = frames
        self.vel = vel
        self.n_clss = n_clss
        self.ch = 4
        self.ch1, self.ch2 = 32, 64

        self.data_bn = nn.BatchNorm1d(self.ch * nodes)
        bn_init(self.data_bn, 1)
        self.drop = nn.Dropout(0.25)
        A = np.stack([np.eye(nodes)] * 3, axis=0)

        if frames:
            self.conv0 = nn.Sequential(
                nn.Conv2d(self.ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.ch1), nn.SiLU())
        if vel:
            self.v0 = nn.Sequential(
                nn.Conv1d(2, self.ch1, 3, bias=False), nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        self.l1 = TCN_GCN_unit(self.ch, self.ch1, A, residual=False)

        if frames:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.ch1), nn.SiLU())
        if vel:
            self.v1 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch1, 3, bias=False),
                nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        self.l2 = TCN_GCN_unit(self.ch1, self.ch2, A)

        if frames:
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch2, kernel_size=2, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.ch2), nn.SiLU())

        if vel:
            self.v2 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False),
                nn.BatchNorm1d(self.ch2), nn.SiLU())

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.BatchNorm1d(self.ch2),
            nn.Sigmoid()
        )

        self.linear = nn.Linear(self.ch2, self.n_clss)
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / self.n_clss))
        self.pool_sig_2d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        if vel:
            self.pool_sig_1d = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Sigmoid()
            )

    def forward(self, kp, frame=None, vel=None):

        N, C, T, V = kp.shape
        kp = kp.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        kp = self.data_bn(kp)
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()  # [2, 4, T, 19]

        if self.frames:
            f1 = self.conv0(frame)  # [2, 32, 190, 62]
        if self.vel:
            v1 = self.v0(vel)  # [2, 32, T-2]

        x1 = self.l1(kp)
        if self.frames:
            f1 = self.conv1(f1)
            x1.mul(self.pool_sig_2d(f1))
        if self.vel:
            v1 = self.v1(v1)
            x1 = x1.mul(self.pool_sig_1d(v1).unsqueeze(-1))

        x1 = self.l2(x1)
        if self.frames:
            f1 = self.conv2(f1)
            x1 = x1.mul(self.pool_sig_2d(f1))
        if self.vel:
            v1 = self.v2(v1)
            x1 = x1.mul(self.pool_sig_1d(v1).unsqueeze(-1))

        x1 = self.gap(x1).squeeze(-1)
        x1 = x1.squeeze(-1)
        x1 = self.att(x1).mul(x1) + x1
        x1 = self.drop(x1)
        x1 = self.linear(x1)

        return x1

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class conv_residual(nn.Module):
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
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        learn_A = self.DecoupleA.repeat(1, self.out_ch // self.groups, 1, 1)  # learn_A -> [3, 32, 19, 19]
        norm_learn_A = torch.cat([self.L2_norm(learn_A[0:1, ...]),
                                  self.L2_norm(learn_A[1:2, ...]),
                                  self.L2_norm(learn_A[2:3, ...])],
                                 0)
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
        super(Encoder, self).__init__()

        self.norm = nn.LayerNorm(inputs)
        self.layers = EncoderLayer(inputs, heads, hidden, a_dropout=a_dropout, f_dropout=f_dropout)

    def forward(self, x, mask=None):
        x = self.layers(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, inputs, heads, hidden, a_dropout=None, f_dropout=None):
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
        gcn = self.gcn(x)
        res = self.residual(x)
        x = gcn + res
        B, C, T, V = x.size()
        tcn = self.embed(x.permute(0, 3, 2, 1)).contiguous().view(B*V, T, -1)
        for i in range(self.tat_times):
            memory = tcn
            tcn = self.tat[i](tcn)
            tcn += memory
        y = self.linear(tcn).contiguous().view(B, -1, T, V)
        y = self.relu(y)
        return y

class CrossTransformer(nn.Module):
    def __init__(self, inputs):
        super(CrossTransformer, self).__init__()
        self.heads = 8
        self.attention = MultiHeadAttention(self.heads, inputs)
        self.mlp = nn.Sequential(
            nn.Linear(1, inputs), nn.ReLU(),
            nn.Linear(inputs, inputs), nn.ReLU(),
            nn.Linear(inputs, 19), nn.ReLU()
        )# k, v -> [..., 19, 32]

    def forward(self, x, y):
        #x=[B, C, T, V]
        #y=[B, C, T]
        B, C, T, V = x.size()
        p = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, -1) # x->[B*T, V, C]
        y = y.permute(0, 2, 1)
        y = y.contiguous().view(B * T, C)
        y = y.unsqueeze(-2) # y->[B*T, 1, C]
        q = self.mlp(y.transpose(-1, -2)).transpose(-1, -2)
        y = y.repeat(1, V, 1) + q
        z = self.attention(p, y, y).view(B, T, V, -1).permute(0, 3, 1, 2)
        z += x
        return z

class FC(nn.Module):
    def __init__(self, dims, activation=None, dropout=None):
        super(FC, self).__init__()
        self.hidden = dims * 3
        #self.conv = nn.Sequential(nn.Conv2d(dims, self.hidden, kernel_size=1, stride=1),
        #    nn.BatchNorm2d(self.hidden), nn.SiLU(),
        #    nn.Conv2d(self.hidden, dims, kernel_size=1, stride=1),
        #    nn.BatchNorm2d(dims), nn.ReLU()
        #)
        #self.linear = nn.Linear(dims, dims)#way1
        self.layers = nn.Sequential(
            nn.Linear(dims, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, dims), nn.ReLU()
        )#way2
        self.bn = nn.BatchNorm2d(dims)
        self.act = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, x):
        #x = self.conv(x)
        #y = self.act(self.bn(x))
        #x = self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)#way1

        x = self.layers(x.permute(0, 2, 3, 1))
        y = self.act(self.bn(x.permute(0, 3, 1, 2)))
        y = self.dropout(y)
        return y

class GatedFusion(nn.Module):
    def __init__(self, dims):
        super(GatedFusion, self).__init__()
        self.fc1 = FC(dims)
        self.fc2 = FC(dims)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        z1 = self.fc1(x)
        z2 = self.fc2(y)
        #z = self.sig(torch.add(z1, z2))
        z = 0
        result = torch.add(x.mul(1 - z), y.mul(z))
        return result
