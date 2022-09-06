import torch
from torch import nn
import numpy as np
import math

from model0906.Multihead_attention import Encoder, Embedding_module


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class ConvResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(ConvResidual, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        y = self.conv(x)
        return self.bn(y)


class DecouplingGcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive) -> None:
        super(DecouplingGcn, self).__init__()
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
            in_channels, out_channels * self.subnet, requires_grad=True, device='cuda'), requires_grad=True)#, device='cuda'
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * self.subnet)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * self.subnet, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)#, device='cuda'
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


class GCN_TAT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(GCN_TAT_unit, self).__init__()
        self.feature_dims = out_channels * 3
        self.num_heads = 8
        self.hidden_dims = out_channels * 6
        self.attention_dropout = 0.3
        self.tat_times = 5

        self.res = residual
        self.gcn = DecouplingGcn(in_channels, out_channels, A, adaptive)

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
            self.residual = ConvResidual(in_channels, out_channels, kernel_size=1, stride=stride)

        self.linear = nn.Linear(self.feature_dims, out_channels)

    def forward(self, x):  # x->[2, 4, T, 19]
        gcn = self.gcn(x)
        res = self.residual(x)
        x = gcn + res
        B, C, T, V = x.size()
        tcn = self.embed(x.permute(0, 3, 2, 1)).contiguous().view(B * V, T, -1)
        for i in range(self.tat_times):
            memory = tcn
            tcn = self.tat[i](tcn)
            tcn += memory
        y = self.linear(tcn).contiguous().view(B, -1, T, V)
        y = self.relu(y)
        return y
