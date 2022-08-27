import torch
from torch import nn

class FC(nn.Module):
    def __init__(self, dims, activation=None, dropout=None):
        super(FC, self).__init__()

        self.conv = nn.Conv2d(dims, dims, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(dims)
        self.act = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        y = self.act(self.bn(x))
        return y

class GatedFusion(nn.Module):
    def __init__(self, dims):
        super(GatedFusion, self).__init__()
        self.layers = 2

        self.fc1 = FC(dims)
        self.fc2 = FC(dims)
        self.sig = nn.Sigmoid()
        self.layer = nn.ModuleList()
        self.conv = nn.Sequential(
                        nn.Conv2d(dims, dims, kernel_size=1, stride=1),
                        nn.BatchNorm2d(dims), nn.ReLU()
        )
        for _ in range(self.layers):
            self.layer.append(self.conv)

    def forward(self, x, y):
        z1 = self.fc1(x)
        z2 = self.fc2(y)
        z = self.sig(torch.add(z1, z2))
        result = torch.add(z1.mul(z), z2.mul(1 - z))
        for i in range(self.layers):
            result = self.layer[i](result)
        return result
