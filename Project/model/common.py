import math
import torch
from torch import nn


def ConvLayer(in_channels, out_channels, dims=1,
              kernel_size=1, stride=1, padding=0, bias=True):
    if dims == 1:
        layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias)
    elif dims == 2:
        layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias)
    return layer

