import torch
from torch import nn

class FC(nn.Module):
    def __init__(self, dims, activation=None, dropout=None):
        super(FC, self).__init__()
        self.hidden = dims * 3
        #self.conv = nn.Conv2d(dims, dims, kernel_size=1, stride=1)
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
        z = self.sig(torch.add(z1, z2))
        result = torch.add(z1.mul(z), z2.mul(1 - z))
        return result
