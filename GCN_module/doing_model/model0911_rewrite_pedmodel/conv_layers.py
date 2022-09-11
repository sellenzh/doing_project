from torch import nn
import torch


class ImgConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,  padding=0, bias=False):
        super(ImgConv, self).__init__()
        self.layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU()

    def forward(self, x):
        y = self.layer(x)
        y = self.relu(self.bn(y))
        return y


class PoolSig(nn.Module):
    def __init__(self, dims):
        super(PoolSig, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) if dims == 2 else nn.AdaptiveAvgPool1d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.sig(y)
        return y


class ImgLayers(nn.Module):
    def __init__(self, ch_in, ch1, ch2):
        super(ImgLayers, self).__init__()
        self.ch, self.ch1, self.ch2 = ch_in, ch1, ch2
        self.conv1 = ImgConv(self.ch, self.ch1)
        self.conv2 = ImgConv(self.ch1, self.ch1)
        self.conv3 = ImgConv(self.ch1, self.ch2, kernel_size=2)
        self.pool_sig = PoolSig(dims=2)

    def forward(self, img):
        f = self.conv1(img)
        f = self.conv2(f)
        f1 = self.pool_sig(f)
        f2 = self.pool_sig(self.conv3(f))
        
        return f1, f2


class VelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,  padding=0, bias=False):
        super(VelConv, self).__init__()
        self.layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.SiLU()

    def forward(self, x):
        y = self.layer(x)
        y = self.relu(self.bn(y))
        return y


class VelLayers(nn.Module):
    def __init__(self, ch_in, ch1, ch2):
        super(VelLayers, self).__init__()
        self.ch, self.ch1, self.ch2 = ch_in, ch1, ch2
        self.conv1 = VelConv(self.ch, self.ch1)
        self.conv2 = VelConv(self.ch1, self.ch1)
        self.conv3 = VelConv(self.ch1, self.ch2, kernel_size=2)
        self.pool_sig = PoolSig(dims=1)

    def forward(self, vel):
        v = self.conv1(vel)
        v = self.conv2(v)
        v1 = self.pool_sig(v).unsqueeze(-1)
        v2 = self.pool_sig(self.conv3(v)).unsqueeze(-1)
        
        return v1, v2


'''model1 = ImgLayers(4, 32, 64)
model2 = VelLayers(2, 32, 64)

img = torch.randn(size=(16, 4, 192, 64))
vel = torch.randn(size=(16, 2, 62))

y1 = model1(img)
y2 = model2(vel)

print(y1[0].size(), y1[1].size(), y2[0].size(), y2[1].size())'''
