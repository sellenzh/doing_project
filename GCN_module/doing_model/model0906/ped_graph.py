from torch import nn
import numpy as np
import math

from model0906.GCN_TAT_unit import GCN_TAT_unit
from model0906.conv_layers import ConvAndPoolImg, ConvAndPoolVel
from model0906.gated_fusion import GatedFusion


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class PedModel(nn.Module):
    def __init__(self, n_clss=1):
        super(PedModel, self).__init__()
        self.ch, self.ch1, self.ch2 = 4, 32, 64
        self.n_clss = n_clss
        self.nodes = 19
        self.heads = 8
        self.rate = 0.3

        self.data_bn = nn.BatchNorm1d(self.ch * self.nodes)
        bn_init(self.data_bn, 1)
        self.drop = nn.Dropout(0.25)
        A = np.stack([np.eye(self.nodes)] * 3, axis=0)

        self.layer1 = GCN_TAT_unit(self.ch, self.ch1, A, adaptive=False)
        self.layer2 = GCN_TAT_unit(self.ch1, self.ch2, A)

        self.img_conv = ConvAndPoolImg(self.ch, self.ch1, self.ch2)
        self.vel_conv = ConvAndPoolVel(self.ch, self.ch1, self.ch2)

        #self.gate1 = GatedFusion(dims=self.ch1, heads=self.heads, dropout_rate=self.rate)
        #self.gate2 = GatedFusion(dims=self.ch2, heads=self.heads, dropout_rate=self.rate)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.BatchNorm1d(self.ch2),
            nn.Sigmoid()
        )
        self.linear = nn.Linear(self.ch2, self.n_clss)
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / self.n_clss))

    def forward(self, pose, frame, vel):
        N, C, T, V = pose.shape
        kp = pose.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        kp = self.data_bn(kp)
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        img1, img2, fr1, fr2 = self.img_conv(frame, time=T)

        vel1, vel2, ve1, ve2 = self.vel_conv(vel, time=T)

        pose = self.layer1(kp)
        pose = pose.mul(img1).mul(vel1)

        #pose = self.gate1(pose, fr1, ve1)

        pose = self.layer2(pose)
        pose = pose.mul(img2).mul(vel2)

        #pose = self.gate2(pose, fr2, ve2)

        pose = self.gap(pose).squeeze(-1)
        pose = pose.squeeze(-1)
        y = self.att(pose).mul(pose) + pose
        y = self.drop(y)
        y = self.linear(y)

        return y

'''import random
import torch
T = random.randint(2, 62)
pose = torch.randn(size=(16, 4, T, 19))
frame = torch.randn(size=(16, 4, 192, 64))
vel = torch.randn(size=(16, 2, 62))
model = PedModel(n_clss=3)
y = model(pose, frame, vel)
print(y.shape)'''
