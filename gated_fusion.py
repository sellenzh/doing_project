import torch
from torch import nn

from model0906.Multihead_attention import Encoder


class GatedFusion(nn.Module):
    def __init__(self, dims, heads, dropout_rate):
        super(GatedFusion, self).__init__()
        self.dims = dims
        self.heads = heads
        self.hidden = self.dims * 3
        self.dropout = dropout_rate
        self.layers = 5

        self.cross_img = CrossAttention(self.dims, self.heads, self.hidden, self.layers, self.dropout)
        self.cross_vel = CrossAttention(self.dims, self.heads, self.hidden, self.layers, self.dropout)

        self.gated_img = Gated(self.dims)
        self.gated_vel = Gated(self.dims)

    def forward(self, pose, img, vel):
        pose_img = self.cross_img(pose, img)
        pose_vel = self.cross_vel(pose, vel)

        pose_image = self.gated_img(pose, pose_img)
        pose_img_vel = self.gated_vel(pose_image, pose_vel)
        return pose_img_vel

class CrossAttention(nn.Module):
    def __init__(self, dims, heads, hidden, layers, dropout=None):
        super(CrossAttention, self).__init__()
        self.heads = heads
        self.layers_num = layers

        self.attention = nn.ModuleList()
        for _ in range(self.layers_num):
            self.attention.append(Encoder(inputs=dims, heads=self.heads, hidden=hidden, a_dropout=dropout, f_dropout=dropout))

    def forward(self, pose, info):
        N, C, T, V = pose.size()
        pose = pose.permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)
        info = info.permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)

        for i in range(self.layers_num):
            pose_att = self.attention[i](pose, info)
            pose_att += pose
        return pose_att.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()

class Gated(nn.Module):
    def __init__(self, dims):
        super(Gated, self).__init__()
        self.layer1 = FC(dims=dims)
        self.layer2 = FC(dims=dims)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        z1 = self.layer1(x)
        z2 = self.layer2(y)
        z = torch.sigmoid(torch.add(z1, z2))
        z = 0
        res = torch.add(torch.mul(x, 1 - z), torch.mul(y, z))
        return res

class FC(nn.Module):
    def __init__(self, dims, activation=None, dropout=None):
        super(FC, self).__init__()
        self.hidden = dims * 3
        self.layers = nn.Sequential(
            nn.Linear(dims, self.hidden),
            nn.LayerNorm(self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.LayerNorm(self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, dims),
            nn.LayerNorm(dims), nn.ReLU()
        )#way2
        self.bn = nn.BatchNorm2d(dims)
        self.act = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, x):
        x = self.layers(x.permute(0, 2, 3, 1))
        y = self.act(self.bn(x.permute(0, 3, 1, 2)))
        y = self.dropout(y)
        return y
