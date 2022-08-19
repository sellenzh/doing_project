'''
updated 22-08-19 10:14(UTC+8)

'''

import math
from math import sqrt
import numpy as np
import torch
from torch import nn

from encoder_module import Encoder

class pedMondel(nn.Module):
    '''
    the model only use pose and vel data.
    '''
    def __init__(self, frames=True, vel=False, seg=False, h3d=True, nodes=19, n_clss=1):
        super(pedMondel, self).__init__()

        self.ch, self.ch1, self.ch2 = 4, 32, 64
        A = np.stack([np.stack([np.eye(nodes)]*3, axis=0)]*5, axis=0)
        #B = np.stack([np.stack([np.eye(nodes)] * 3, axis=0).unsqueeze(0)] * 5, axis=0)

        self.data_bn = nn.BatchNorm1d(self.ch * nodes)
        bn_init(self.data_bn, 1)

        self.drop = nn.Dropout(0.25)
        self.pose_layer1 = GCN_TAT_unit(self.ch, self.ch1, A, residual=False)
        self.pose_layer2 = GCN_TAT_unit(self.ch1, self.ch2, A)

        self.img_layer1 = nn.Sequential(
            nn.Conv2d(self.ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.SiLU()
        )
        self.img_layer2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.SiLU()
        )
        self.img_layer3 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch2), nn.SiLU()
        )

        self.vel_layer1 = nn.Sequential(
            nn.Conv1d(2, self.ch1, kernel_size=3, bias=False),
            nn.BatchNorm1d(self.ch1), nn.SiLU()
        )
        self.vel_layer2 = nn.Sequential(
            nn.Conv1d(self.ch1, self.ch1, kernel_size=3, bias=False),
            nn.BatchNorm1d(self.ch1), nn.SiLU()
        )
        self.vel_layer3 = nn.Sequential(
            nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False),
            nn.BatchNorm1d(self.ch2), nn.SiLU()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(nn.SiLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.BatchNorm1d(self.ch2), nn.Sigmoid()
        )
        self.linear = nn.Linear(self.ch2, n_clss)
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / n_clss))

        self.pool_sig_2d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid()
        )
        self.pool_sig_1d = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Sigmoid()
        )
        #---------------
        #self.pool_sig_2d_2 = nn.Sequential(
        #    nn.AdaptiveAvgPool2d(1), nn.Sigmoid()
        #)
        #self.conv = nn.Conv2d(self.ch1, self.ch2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, kp, frames=None, vel=None):
        n, c, t, v = kp.shape
        kp = kp.permute(0, 1, 3, 2).contiguous().view(n, c*v, t)
        kp = self.data_bn(kp)
        kp = kp.contiguous().view(n, c, v, t).permute(0, 1, 3, 2).contiguous()

        # 1st key_points stage.
        x = self.pose_layer1(kp)

        # images processed by 2 Convolution 2d.
        f = self.img_layer2(self.img_layer1(frames))
        # pool images to weight at each channel, channel-wise to pose.
        x.mul(self.pool_sig_2d(f))
        # velocity processed by 2 Convolution 1d.
        v = self.vel_layer2(self.vel_layer1(vel))
        # pool velocity to weight at each channel, channel-wise to pose.
        x.mul(self.pool_sig_1d(v).unsqueeze(-1))

        # 2nd key_points stage.
        x = self.pose_layer2(x)

        f1 = self.img_layer3(f)
        x.mul(self.pool_sig_2d(f1))
        v1 = self.vel_layer3(v)
        x.mul(self.pool_sig_1d(v1).unsqueeze(-1))

        y = self.gap(x).squeeze(-1)
        y = y.squeeze(-1)
        y = self.att(y).mul(y) + y
        y = self.drop(y)
        y = self.linear(y)
        return y


class decoupling_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, adaptive):
        super(decoupling_gcn, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.subnet = A.shape[0]
        self.groups = groups

        self.linear = nn.Linear(self.in_ch, self.out_ch)
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32),
                        [3, 1, 19, 19]), dtype=torch.float32, requires_grad=True).repeat(1, self.groups, 1, 1), requires_grad=True)

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

        self.linear_weight = nn.Parameter(
            torch.zeros(in_channels, out_channels * self.subnet,
            requires_grad=True), requires_grad=True)
        nn.init.normal_(self.linear_weight, 0, math.sqrt(
            0.5 / (out_channels * self.subnet)
        ))
        self.linear_bias = nn.Parameter(
            torch.zeros(1, out_channels * self.subnet, 1, 1,
            requires_grad=True), requires_grad=True)
        nn.init.constant(self.linear_bias, 1e-6)

    def L2_norm(selfself, A):
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4
        A = A / A_norm
        return A

    def forward(self, x):
        learn_A = self.DecoupleA.repeat(1, self.out_ch // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.L2_norm(learn_A[0:1, ...]),
                                  self.L2_norm(learn_A[1:2, ...]),
                                  self.L2_norm(learn_A[2:3, ...])], 0)
        y = torch.einsum('nctw,cd->ndtw', (x, self.linear_weight)).contiguous()
        y += self.linear_bias
        y = self.bn0(y)

        n, kc, t, v = y.size()
        y = y.view(n, self.subnet, kc // self.subnet, t, v)
        y = torch.einsum('nkctv,kcvw->nctw', (y, norm_learn_A))

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y

class conv_residual(nn.Module):
    '''the residual of gcn and temporal attention module
        to adjust channels by using convolution
    '''
    def __init__(self, in_channels, out_channels,kernel_size=1, stride=1):
        super(conv_residual, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))

class GCN_TAT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, groups=8, residual=True, adaptive=True):
        super(GCN_TAT_unit, self).__init__()
        self.feature_dims = 128
        self.num_heads = 8
        self.hidden_dims = 256
        self.attention_dropout = 0.3
        self.tat_times = 5
        self.gcn_times = 5
        self.groups = groups
        self.res = residual

        self.gcn = nn.ModuleList()
        self.gcn.append(decoupling_gcn(in_channels, out_channels, A[0], groups=groups, adaptive=adaptive))
        for i in range(self.gcn_times-1):
            self.gcn.append(decoupling_gcn(self.feature_dims, self.feature_dims, A[i+1], groups=groups, adaptive=adaptive))
        self.embed = Embedding_module(out_channels, self.feature_dims)
        self.tcn = nn.ModuleList(Encoder(self.feature_dims, self.num_heads, self.hidden_dims, self.attention_dropout) for _ in range(self.tat_times))

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = conv_residual(in_channels, out_channels, kernel_size=1, stride=stride)

        self.linear = nn.Linear(self.feature_dims, out_channels)

    def forward(self, x):
        for i in range(self.gcn_times):
            gcn = self.gcn[i](x)
            if i == 0:
                res = self.residual(x)
            elif self.res:
                res = x
            else:
                res = 0
            x = gcn + res
        for i in range(self.tat_times):
            b, c, t, v = x.size()
            tcn = self.embed(x.permute(0, 3, 2, 1)).contiguous().view(b*v, t, -1)\
                if i == 0 else x.permute(0, 3, 2, 1).contiguous().view(b*v, t, -1)
            tcn += self.tcn[i](tcn)
            x = tcn.contiguous().view(b, v, t, -1).permute(0, 3, 2, 1)\
                if i != self.tat_times-1 else tcn.contiguous().view(b, v, t, -1)
        y = self.linear(x).permute(0, 3, 2, 1)
        return self.relu(y)


def conv_init(layer):
    if layer.weight is not None:
        nn.init.kaiming_normal_(layer.weight, mode='fan_out')
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

def bn_init(layer, scale):
    nn.init.constant_(layer.weight, scale)
    nn.init.constant_(layer.bias, 0)

class Embedding_module(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Embedding_module, self).__init__()
        self.out_ch = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        return self.linear(x) / sqrt(self.out_ch)

'''
import random
T = random.randint(2, 62)
img = torch.randn(size=(16, 4, 164, 92))
pose = torch.randn(size=(16, 4, T, 19))
model = pedMondel()
y = model(kp=pose, frames=img, vel=None)
print(y.size())'''