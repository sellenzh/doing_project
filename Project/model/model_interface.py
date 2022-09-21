import torch
from torch import nn
import numpy as np


class PedModel(nn.Module):
    def __init__(self, n_clss=1):
        super().__init__()
        self.nodes = 19
        self.n_clss = n_clss
        self.ch, self.ch1, self.ch2 = 4, 32, 64

        self.data_bn = nn.BatchNorm1d(self.ch * self.nodes)
        bn_init(self.data_bn, 1)
        self.drop = nn.Dropout(0.3)
        A = np.stack([np.eye(self.nodes)] * 3, axis=0)

        self.img1 = nn.Sequential(
            nn.Conv2d(self.ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.SiLU()
        )
        self.vel1 = nn.Sequential(
            nn.Conv1d(2, self.ch1, kernel_size=9, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.ch1), nn.SiLU()
        )




    def forward(self, kp, frame, vel):




def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
