import math
import torch
from torch import nn
import numpy as np
from math import sqrt

class pedMondel(nn.Module):
    def __init__(self, frames=True, vel=False, seg=False, h3d=True, nodes=19, n_clss=1):
        super(pedMondel, self).__init__()

        self.ch, self.ch1, self.ch2 = 4, 32, 64

        self.drop = nn.Dropout(0.25)

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

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(nn.SiLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.BatchNorm1d(self.ch2), nn.Sigmoid()
        )
        self.linear = nn.Linear(self.ch2, n_clss)
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / n_clss))

        self.pool_sig_2d_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid()
        )
        self.pool_sig_2d_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid()
        )
        self.conv = nn.Conv2d(self.ch1, self.ch2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, kp, frames=None, vel=None):
        f = self.img_layer2(self.img_layer1(frames))
        f1 = self.conv(self.pool_sig_2d_1(f))
        y = self.pool_sig_2d_2(self.img_layer3(f)).mul(f1).squeeze()
        y += self.att(y).mul(y)
        y = self.linear(self.drop(y))
        return y
'''
tensor = torch.randn(size=(16, 4, 164, 92))
model = pedMondel()
y = model(kp=tensor, frames=tensor, vel=tensor)
print(y.size())'''