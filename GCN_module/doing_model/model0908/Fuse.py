import torch
from torch import nn

class UAFM(nn.Module):
    def __init__(self, dims):
        super(UAFM, self).__init__()
        self.conv_x = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dims), nn.ReLU()
        )
        self.conv_y = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dims), nn.ReLU()
        )

        self.fuse = nn.Conv2d(dims * 2, dims, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(dims)
        self.relu = nn.ReLU()
        self.out = nn.Conv2d(dims, dims, kernel_size=1, padding=0, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv_x(x)
        y = self.conv_y(y)

        cat = torch.cat((x, y), dim=1)
        weight = self.fuse(cat)
        weight = self.relu(self.bn(weight))

        weight = self.sig(self.out(weight))

        return weight

from torch import nn
class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super(ECA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sig(y)
        return y #return [batch_size, channels, 1, 1] size of weight.
        #return x * y.expand_as(x) # return size equal to input.


import torch
from torch import nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    def __init__(self, dims):
        super(CCA, self).__init__()

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(dims, dims, bias=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        channel_att_x = self.mlp(avg_pool_x)
        scale = torch.sigmoid(channel_att_x).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        y = self.relu(x_after_channel)
        return y